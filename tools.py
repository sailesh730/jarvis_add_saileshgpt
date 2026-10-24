"""General and travel tools used by the LangGraph agents."""

import ast
import ipaddress
import math
import os
import socket
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from rag import search_uploaded_document

load_dotenv()

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
MEMORY_DB = DATA_DIR / "chat_memory.sqlite"

SYSTEM_PROMPT = """You are SaileshGPT's travel assistant.

PLANNING WORKFLOW
- For trip plans, establish the origin, destination, dates or trip length, travelers and budget.
  If a detail is missing, make a reasonable assumption, state it, and continue unless it prevents
  a useful answer.
- Call get_today when dates or date-sensitive recommendations matter.
- Use get_country_info and get_main_airports to research countries and identify likely airports.
- Use search_airports when an airport or city needs verification.
- Use search_flights_by_route or list_airlines_on_route for route information, and get_flight_status
  or get_airport_board only when the user asks about a specific flight or current airport operations.
- Use get_weather_forecast for destination conditions during the trip; use get_weather for current
  conditions.
- Use web search, when available, for current visa and entry rules, attractions, accommodation
  areas, local transport, safety advisories and typical prices. Prefer official sources for rules.
- Convert costs with convert_currency when needed, then use estimate_trip_budget for totals.
- If tools are unavailable or fail, say what could not be verified and continue with clearly
  labeled estimates or general guidance. Never invent tool results.

TRIP PLAN RESPONSE
- Give a short overview of the route and airports.
- Include visa and entry notes with a reminder to verify requirements on official sources.
- Provide a practical day-by-day itinerary when the trip length allows.
- Include weather and packing tips when forecast data is available.
- Show budget estimates and state the currency and assumptions.
- Add relevant transport, connectivity, safety and local-etiquette tips.

ACCURACY
- Flight data is live or recent operational information only; it is not fares or far-future
  schedules. Tell the user to check airlines or booking sites for exact times and prices.
- Label estimates clearly. Cite or link sources returned by web search when available.
- Never claim a tool was used unless you called it, and keep replies clear and concise."""


def _calculate(node: ast.AST):
    if isinstance(node, ast.Expression):
        return _calculate(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _calculate(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left = _calculate(node.left)
        right = _calculate(node.right)
        operations = {
            ast.Add: lambda: left + right,
            ast.Sub: lambda: left - right,
            ast.Mult: lambda: left * right,
            ast.Div: lambda: left / right,
            ast.FloorDiv: lambda: left // right,
            ast.Mod: lambda: left % right,
            ast.Pow: lambda: left**right,
        }
        operation = operations.get(type(node.op))
        if operation is None:
            raise ValueError("Only basic arithmetic operators are supported.")
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("Exponent is too large.")
        return operation()
    raise ValueError("Use numbers and basic arithmetic operators only.")


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression accurately."""
    if len(expression) > 256:
        return "Calculation failed: expression is too long."
    try:
        result = _calculate(ast.parse(expression, mode="eval"))
        if not math.isfinite(result):
            return "Calculation failed: result is not finite."
        return str(result)
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError, OverflowError) as exc:
        return f"Calculation failed: {exc}"


@tool
def get_today() -> str:
    """Get today's date."""
    return datetime.now().astimezone().strftime("%A, %d %B %Y")


@tool
def get_weather(location: str) -> str:
    """Get current weather conditions for a city or place name."""
    try:
        geocode = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": location, "count": 1, "language": "en", "format": "json"},
            timeout=10,
        )
        geocode.raise_for_status()
        places = geocode.json().get("results") or []
        if not places:
            return f"Weather lookup failed: no location found for {location!r}."
        place = places[0]
        forecast = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m",
                "timezone": "auto",
            },
            timeout=10,
        )
        forecast.raise_for_status()
        current = forecast.json()["current"]
        return (
            f"Current conditions for {place['name']}, {place.get('country', '')}: "
            f"{current['temperature_2m']}°C, feels like {current['apparent_temperature']}°C, "
            f"humidity {current['relative_humidity_2m']}%, wind "
            f"{current['wind_speed_10m']} km/h (weather code {current['weather_code']})."
        )
    except (requests.RequestException, KeyError, ValueError) as exc:
        return f"Weather lookup failed: {exc}"


@tool
def browser_read(url: str) -> str:
    """Read the visible text from a public HTTP or HTTPS web page."""
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username:
        return "Page read failed: provide a public HTTP or HTTPS URL."
    try:
        addresses = {
            ipaddress.ip_address(record[4][0])
            for record in socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
        }
        if not addresses or any(not address.is_global for address in addresses):
            return "Page read failed: private or local network addresses are not allowed."
        response = requests.get(
            url,
            timeout=15,
            headers={"User-Agent": "SaileshGPT/1.0"},
            allow_redirects=False,
            stream=True,
        )
        if 300 <= response.status_code < 400:
            return "Page read failed: redirects are not followed; provide the destination URL directly."
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").lower()
        if not any(kind in content_type for kind in ("text/html", "text/plain", "application/xhtml")):
            return "Page read failed: the URL does not return an HTML or text page."
        chunks = []
        size = 0
        for chunk in response.iter_content(chunk_size=8192):
            size += len(chunk)
            if size > 1_000_000:
                return "Page read failed: page is larger than the 1 MB limit."
            chunks.append(chunk)
        soup = BeautifulSoup(b"".join(chunks), "html.parser")
        for element in soup(["script", "style", "noscript", "svg"]):
            element.decompose()
        text = " ".join(soup.get_text(" ").split())
        return text[:12_000] or "The page did not contain readable text."
    except (requests.RequestException, socket.gaierror, ValueError) as exc:
        return f"Page read failed: {exc}"


def _thread_id(config: RunnableConfig) -> str:
    configurable = config.get("configurable") or {}
    thread_id = configurable.get("thread_id")
    if not isinstance(thread_id, str) or not thread_id:
        raise ValueError("Conversation context is unavailable.")
    return thread_id


def _open_memory_db() -> sqlite3.Connection:
    connection = sqlite3.connect(MEMORY_DB, timeout=10)
    connection.execute(
        "CREATE TABLE IF NOT EXISTS memories ("
        "thread_id TEXT NOT NULL, note TEXT NOT NULL, created_at TEXT NOT NULL)"
    )
    return connection


@tool
def remember_this(note: str, config: RunnableConfig) -> str:
    """Remember a short detail for this conversation when the user asks."""
    note = note.strip()
    if not note or len(note) > 2000:
        return "Memory was not saved: provide a note of 1 to 2000 characters."
    try:
        with _open_memory_db() as connection:
            connection.execute(
                "INSERT INTO memories (thread_id, note, created_at) VALUES (?, ?, ?)",
                (_thread_id(config), note, datetime.now().astimezone().isoformat()),
            )
        return "Saved that detail for this conversation."
    except (sqlite3.Error, ValueError) as exc:
        return f"Memory was not saved: {exc}"


@tool
def recall_memory(config: RunnableConfig) -> str:
    """Recall details previously saved in this conversation."""
    try:
        with _open_memory_db() as connection:
            rows = connection.execute(
                "SELECT note, created_at FROM memories "
                "WHERE thread_id = ? ORDER BY rowid DESC LIMIT 20",
                (_thread_id(config),),
            ).fetchall()
        if not rows:
            return "No details have been saved for this conversation."
        return "\n".join(f"- {note} (saved {created_at})" for note, created_at in rows)
    except (sqlite3.Error, ValueError) as exc:
        return f"Memory could not be recalled: {exc}"


AVIATIONSTACK_KEY = os.getenv("AVIATIONSTACK_API_KEY")
AVIATIONSTACK_URL = "https://api.aviationstack.com/v1"
RESTCOUNTRIES_URL = "https://restcountries.com/v3.1"
OPEN_METEO_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
EXCHANGE_RATES_URL = "https://open.er-api.com/v6/latest"
VALID_FLIGHT_STATUSES = {"scheduled", "active", "landed", "cancelled", "incident", "diverted"}

COUNTRY_ALIASES = {
    "uae": "AE",
    "emirates": "AE",
    "uk": "GB",
    "britain": "GB",
    "england": "GB",
    "usa": "US",
    "america": "US",
    "south korea": "KR",
    "holland": "NL",
}

REGIONAL_BLOCS = {
    "GCC": {"BH", "KW", "OM", "QA", "SA", "AE"},
    "SAARC": {"AF", "BD", "BT", "IN", "MV", "NP", "PK", "LK"},
    "BIMSTEC": {"BD", "BT", "IN", "MM", "NP", "LK", "TH"},
    "ASEAN": {"BN", "KH", "ID", "LA", "MY", "MM", "PH", "SG", "TH", "VN", "TL"},
}

MAIN_AIRPORTS = {
    "NP": [
        ("KTM", "Tribhuvan International Airport", "Kathmandu"),
        ("PKR", "Pokhara International Airport", "Pokhara"),
        ("BWA", "Gautam Buddha International Airport", "Bhairahawa"),
        ("BIR", "Biratnagar Airport", "Biratnagar"),
    ],
    "AE": [
        ("DXB", "Dubai International Airport", "Dubai"),
        ("AUH", "Zayed International Airport", "Abu Dhabi"),
        ("SHJ", "Sharjah International Airport", "Sharjah"),
        ("DWC", "Al Maktoum International Airport", "Dubai South"),
        ("RKT", "Ras Al Khaimah International Airport", "Ras Al Khaimah"),
        ("FJR", "Fujairah International Airport", "Fujairah"),
    ],
    "IN": [
        ("DEL", "Indira Gandhi International Airport", "Delhi"),
        ("BOM", "Chhatrapati Shivaji Maharaj International Airport", "Mumbai"),
        ("BLR", "Kempegowda International Airport", "Bengaluru"),
        ("MAA", "Chennai International Airport", "Chennai"),
        ("CCU", "Netaji Subhas Chandra Bose International Airport", "Kolkata"),
        ("HYD", "Rajiv Gandhi International Airport", "Hyderabad"),
    ],
    "US": [
        ("JFK", "John F. Kennedy International Airport", "New York"),
        ("LAX", "Los Angeles International Airport", "Los Angeles"),
        ("ORD", "O'Hare International Airport", "Chicago"),
        ("ATL", "Hartsfield-Jackson Atlanta International Airport", "Atlanta"),
        ("SFO", "San Francisco International Airport", "San Francisco"),
        ("DFW", "Dallas/Fort Worth International Airport", "Dallas"),
    ],
    "GB": [
        ("LHR", "Heathrow Airport", "London"),
        ("LGW", "Gatwick Airport", "London"),
        ("STN", "Stansted Airport", "London"),
        ("MAN", "Manchester Airport", "Manchester"),
        ("EDI", "Edinburgh Airport", "Edinburgh"),
    ],
    "QA": [("DOH", "Hamad International Airport", "Doha")],
    "SA": [
        ("RUH", "King Khalid International Airport", "Riyadh"),
        ("JED", "King Abdulaziz International Airport", "Jeddah"),
        ("DMM", "King Fahd International Airport", "Dammam"),
    ],
    "KW": [("KWI", "Kuwait International Airport", "Kuwait City")],
    "OM": [("MCT", "Muscat International Airport", "Muscat")],
    "BH": [("BAH", "Bahrain International Airport", "Manama")],
    "TH": [
        ("BKK", "Suvarnabhumi Airport", "Bangkok"),
        ("DMK", "Don Mueang International Airport", "Bangkok"),
        ("HKT", "Phuket International Airport", "Phuket"),
        ("CNX", "Chiang Mai International Airport", "Chiang Mai"),
    ],
    "SG": [("SIN", "Changi Airport", "Singapore")],
    "MY": [
        ("KUL", "Kuala Lumpur International Airport", "Kuala Lumpur"),
        ("PEN", "Penang International Airport", "Penang"),
    ],
    "ID": [
        ("CGK", "Soekarno-Hatta International Airport", "Jakarta"),
        ("DPS", "Ngurah Rai International Airport", "Bali"),
    ],
    "JP": [
        ("HND", "Haneda Airport", "Tokyo"),
        ("NRT", "Narita International Airport", "Tokyo"),
        ("KIX", "Kansai International Airport", "Osaka"),
    ],
    "KR": [
        ("ICN", "Incheon International Airport", "Seoul"),
        ("GMP", "Gimpo International Airport", "Seoul"),
    ],
    "CN": [
        ("PEK", "Beijing Capital International Airport", "Beijing"),
        ("PKX", "Beijing Daxing International Airport", "Beijing"),
        ("PVG", "Shanghai Pudong International Airport", "Shanghai"),
        ("CAN", "Guangzhou Baiyun International Airport", "Guangzhou"),
    ],
    "TR": [
        ("IST", "Istanbul Airport", "Istanbul"),
        ("SAW", "Sabiha Gokcen International Airport", "Istanbul"),
        ("AYT", "Antalya Airport", "Antalya"),
    ],
    "FR": [
        ("CDG", "Charles de Gaulle Airport", "Paris"),
        ("ORY", "Orly Airport", "Paris"),
        ("NCE", "Nice Cote d'Azur Airport", "Nice"),
    ],
    "DE": [
        ("FRA", "Frankfurt Airport", "Frankfurt"),
        ("MUC", "Munich Airport", "Munich"),
        ("BER", "Berlin Brandenburg Airport", "Berlin"),
    ],
    "AU": [
        ("SYD", "Sydney Kingsford Smith Airport", "Sydney"),
        ("MEL", "Melbourne Airport", "Melbourne"),
        ("BNE", "Brisbane Airport", "Brisbane"),
    ],
    "CA": [
        ("YYZ", "Toronto Pearson International Airport", "Toronto"),
        ("YVR", "Vancouver International Airport", "Vancouver"),
        ("YUL", "Montreal-Trudeau International Airport", "Montreal"),
    ],
    "BD": [("DAC", "Hazrat Shahjalal International Airport", "Dhaka")],
    "LK": [("CMB", "Bandaranaike International Airport", "Colombo")],
    "MV": [("MLE", "Velana International Airport", "Male")],
    "BT": [("PBH", "Paro International Airport", "Paro")],
    "PK": [
        ("KHI", "Jinnah International Airport", "Karachi"),
        ("LHE", "Allama Iqbal International Airport", "Lahore"),
        ("ISB", "Islamabad International Airport", "Islamabad"),
    ],
    "EG": [("CAI", "Cairo International Airport", "Cairo")],
}

WMO_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Rime fog",
    51: "Light drizzle",
    53: "Drizzle",
    55: "Heavy drizzle",
    61: "Light rain",
    63: "Rain",
    65: "Heavy rain",
    71: "Light snow",
    73: "Snow",
    75: "Heavy snow",
    80: "Rain showers",
    81: "Rain showers",
    82: "Violent rain showers",
    95: "Thunderstorm",
    96: "Thunderstorm with hail",
    99: "Thunderstorm with heavy hail",
}


def _get_json(url: str, params: Optional[dict] = None, check_status: bool = True) -> Any:
    try:
        response = requests.get(url, params=params, timeout=15)
        if check_status:
            response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout as exc:
        raise RuntimeError("The request timed out. Please try again.") from exc
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        raise RuntimeError(f"HTTP {status} error from {url}") from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(f"Network error: {exc}") from exc
    except ValueError as exc:
        raise RuntimeError("The service returned an invalid response.") from exc


def _aviation(endpoint: str, params: Optional[dict] = None) -> dict:
    if not AVIATIONSTACK_KEY:
        raise RuntimeError("AVIATIONSTACK_API_KEY is not set in your .env file.")
    query = {"access_key": AVIATIONSTACK_KEY}
    query.update({key: value for key, value in (params or {}).items() if value not in (None, "")})
    data = _get_json(f"{AVIATIONSTACK_URL}/{endpoint}", query, check_status=False)
    if not isinstance(data, dict):
        raise RuntimeError("Aviationstack returned an unexpected response.")
    if "error" in data:
        error = data["error"]
        raise RuntimeError(
            f"Aviationstack error ({error.get('code', 'unknown')}): {error.get('message', '')}"
        )
    return data


def _fetch_country(query: str) -> Optional[dict]:
    country_query = query.strip()
    country_query = COUNTRY_ALIASES.get(country_query.lower(), country_query)
    attempts = []
    if country_query.isalpha() and len(country_query) in (2, 3):
        attempts.append((f"{RESTCOUNTRIES_URL}/alpha/{country_query}", None))
    attempts.extend(
        [
            (f"{RESTCOUNTRIES_URL}/name/{country_query}", {"fullText": "true"}),
            (f"{RESTCOUNTRIES_URL}/name/{country_query}", None),
        ]
    )
    for url, params in attempts:
        try:
            data = _get_json(url, params)
        except RuntimeError as exc:
            if str(exc).startswith("HTTP 404"):
                continue
            raise
        if isinstance(data, list) and data:
            return data[0]
        if isinstance(data, dict) and "name" in data:
            return data
    return None


def _border_names(codes: list) -> list:
    if not codes:
        return []
    try:
        data = _get_json(
            f"{RESTCOUNTRIES_URL}/alpha",
            {"codes": ",".join(codes), "fields": "name,cca3"},
        )
        return sorted(item["name"]["common"] for item in data)
    except (RuntimeError, KeyError, TypeError):
        return codes


def _format_flight(flight: dict) -> str:
    departure = flight.get("departure") or {}
    arrival = flight.get("arrival") or {}
    airline = flight.get("airline") or {}
    flight_info = flight.get("flight") or {}

    def format_time(value: Optional[str]) -> str:
        if not value:
            return "N/A"
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).strftime("%d %b %Y, %H:%M")
        except ValueError:
            return value

    def format_delay(minutes: Optional[int]) -> str:
        if not minutes:
            return "On time"
        hours, remainder = divmod(int(minutes), 60)
        return f"{hours}h {remainder}m delay" if hours else f"{remainder}m delay"

    return (
        f"Flight:  {flight_info.get('iata') or flight_info.get('icao') or 'N/A'}\n"
        f"Airline: {airline.get('name') or 'N/A'}\n"
        f"Status:  {(flight.get('flight_status') or 'unknown').title()}\n"
        f"Date:    {flight.get('flight_date') or 'N/A'}\n"
        "DEPARTURE\n"
        f"  Airport:   {departure.get('airport') or 'N/A'} ({departure.get('iata') or '?'})\n"
        f"  Terminal:  {departure.get('terminal') or 'N/A'} | Gate: {departure.get('gate') or 'N/A'}\n"
        f"  Scheduled: {format_time(departure.get('scheduled'))}\n"
        f"  Estimated: {format_time(departure.get('estimated'))}\n"
        f"  Delay:     {format_delay(departure.get('delay'))}\n"
        "ARRIVAL\n"
        f"  Airport:   {arrival.get('airport') or 'N/A'} ({arrival.get('iata') or '?'})\n"
        f"  Terminal:  {arrival.get('terminal') or 'N/A'} | Gate: {arrival.get('gate') or 'N/A'}\n"
        f"  Scheduled: {format_time(arrival.get('scheduled'))}\n"
        f"  Estimated: {format_time(arrival.get('estimated'))}\n"
        f"  Delay:     {format_delay(arrival.get('delay'))}"
    )


def _format_airport(airport: dict) -> str:
    return (
        f"{airport.get('airport_name', 'N/A')}\n"
        f"  IATA: {airport.get('iata_code', 'N/A')} | ICAO: {airport.get('icao_code', 'N/A')}\n"
        f"  Country: {airport.get('country_name', 'N/A')} ({airport.get('country_iso2', '?')})\n"
        f"  Timezone: {airport.get('timezone', 'N/A')} (GMT {airport.get('gmt', 'N/A')})\n"
        f"  Coordinates: {airport.get('latitude', 'N/A')}, {airport.get('longitude', 'N/A')}"
    )


def _format_airline(airline: dict) -> str:
    return (
        f"{airline.get('airline_name', 'N/A')}\n"
        f"  IATA: {airline.get('iata_code', 'N/A')} | ICAO: {airline.get('icao_code', 'N/A')}\n"
        f"  Country: {airline.get('country_name', 'N/A')} | Hub: {airline.get('hub_code') or 'N/A'}\n"
        f"  Fleet size: {airline.get('fleet_size') or 'N/A'} | Founded: {airline.get('date_founded') or 'N/A'}\n"
        f"  Status: {airline.get('status') or 'N/A'} | Type: {airline.get('type') or 'N/A'}"
    )


@tool
def get_country_info(country: str) -> str:
    """Get a country's codes, capital, languages, currency, neighbours, time zones and regional blocs."""
    try:
        info = _fetch_country(country)
    except RuntimeError as exc:
        return f"Error: {exc}"
    if not info:
        return f"Country '{country}' not found."

    code = info.get("cca2", "")
    languages = ", ".join((info.get("languages") or {}).values()) or "N/A"
    currencies = ", ".join(
        f"{value.get('name')} ({currency_code}, {value.get('symbol', '')})"
        for currency_code, value in (info.get("currencies") or {}).items()
    ) or "N/A"
    calling = info.get("idd") or {}
    suffixes = calling.get("suffixes") or []
    calling_code = calling.get("root", "") + (suffixes[0] if len(suffixes) == 1 else "")
    blocs = [name for name, members in REGIONAL_BLOCS.items() if code in members]
    borders = _border_names(info.get("borders") or [])
    return (
        f"{info['name']['common']} ({info['name'].get('official', '')})\n"
        f"  Codes: {code} / {info.get('cca3', '')}\n"
        f"  Capital: {', '.join(info.get('capital') or ['N/A'])}\n"
        f"  Region: {info.get('region', 'N/A')} - {info.get('subregion', 'N/A')}\n"
        f"  Population: {info.get('population', 0):,} | Area: {info.get('area', 0):,.0f} km2\n"
        f"  Languages: {languages}\n"
        f"  Currency: {currencies}\n"
        f"  Calling code: {calling_code or 'N/A'}\n"
        f"  Time zones: {', '.join(info.get('timezones') or ['N/A'])}\n"
        f"  Neighbouring countries: {', '.join(borders) or 'None (no land borders)'}\n"
        f"  Regional blocs: {', '.join(blocs) or 'None in built-in list'}\n"
        f"  Drives on the: {(info.get('car') or {}).get('side', 'N/A')}"
    )


@tool
def get_main_airports(country: str) -> str:
    """List main airports for a country by IATA code, name and city."""
    try:
        info = _fetch_country(country)
    except RuntimeError as exc:
        return f"Error: {exc}"
    if not info:
        return f"Country '{country}' not found."
    name = info["name"]["common"]
    airports = MAIN_AIRPORTS.get(info.get("cca2"))
    if not airports:
        return f"No built-in airport list for {name}. Use search_airports to look them up."
    return "\n".join(
        [f"Main airports in {name}:"]
        + [f"  {iata} - {airport_name}, {city}" for iata, airport_name, city in airports]
    )


@tool
def search_airports(search: str) -> str:
    """Look up airport details by airport name, city or code (requires Aviationstack API key)."""
    try:
        data = _aviation("airports", {"search": search.strip(), "limit": 5})
    except RuntimeError as exc:
        return f"Error: {exc}"
    airports = data.get("data") or []
    if not airports:
        return f"No airports found for '{search}'."
    return "\n\n".join(_format_airport(airport) for airport in airports)


@tool
def get_flight_status(flight_iata: str) -> str:
    """Get current or recent status, schedule and delay for an IATA flight code."""
    normalized = flight_iata.strip().upper()
    try:
        data = _aviation("flights", {"flight_iata": normalized, "limit": 3})
    except RuntimeError as exc:
        return f"Error: {exc}"
    flights = data.get("data") or []
    if not flights:
        return f"No flight found for {normalized}."
    return "\n" + ("-" * 40 + "\n").join(_format_flight(flight) for flight in flights)


@tool
def search_flights_by_route(
    departure_iata: str,
    arrival_iata: str,
    flight_status: Optional[str] = None,
    limit: int = 5,
) -> str:
    """Find recent or live flights between two airport IATA codes (not fares or future schedules)."""
    status = flight_status.strip().lower() if flight_status else None
    if status and status not in VALID_FLIGHT_STATUSES:
        return f"Invalid flight_status. Choose from: {', '.join(sorted(VALID_FLIGHT_STATUSES))}."
    departure = departure_iata.strip().upper()
    arrival = arrival_iata.strip().upper()
    try:
        data = _aviation(
            "flights",
            {
                "dep_iata": departure,
                "arr_iata": arrival,
                "flight_status": status,
                "limit": max(1, min(limit, 100)),
            },
        )
    except RuntimeError as exc:
        return f"Error: {exc}"
    flights = data.get("data") or []
    if not flights:
        return f"No flights found from {departure} to {arrival}."
    return f"Found {len(flights)} flight(s):\n\n" + ("\n" + "-" * 40 + "\n").join(
        _format_flight(flight) for flight in flights
    )


@tool
def list_airlines_on_route(departure_iata: str, arrival_iata: str) -> str:
    """Show airlines and flight numbers found on a route from recent flight records."""
    departure = departure_iata.strip().upper()
    arrival = arrival_iata.strip().upper()
    try:
        data = _aviation("flights", {"dep_iata": departure, "arr_iata": arrival, "limit": 100})
    except RuntimeError as exc:
        return f"Error: {exc}"
    flights = data.get("data") or []
    if not flights:
        return f"No flights found from {departure} to {arrival}."

    airlines = {}
    for flight in flights:
        name = (flight.get("airline") or {}).get("name") or "Unknown"
        number = (flight.get("flight") or {}).get("iata")
        scheduled = (flight.get("departure") or {}).get("scheduled")
        entry = airlines.setdefault(name, {"flights": set(), "times": set()})
        if number:
            entry["flights"].add(number)
        if scheduled and len(scheduled) >= 16:
            entry["times"].add(scheduled[11:16])
    lines = [f"Airlines seen on {departure} -> {arrival}:"]
    for name, entry in sorted(airlines.items()):
        lines.append(
            f"  - {name}: flights {', '.join(sorted(entry['flights'])) or 'N/A'} | "
            f"typical departures {', '.join(sorted(entry['times'])) or 'N/A'}"
        )
    lines.append("Based on recent flight records; check the airline for exact schedules and fares.")
    return "\n".join(lines)


@tool
def get_airport_board(airport_iata: str, direction: str = "departures", limit: int = 10) -> str:
    """Show recent arrivals or departures for an airport IATA code."""
    normalized_direction = direction.strip().lower()
    if normalized_direction not in {"arrival", "arrivals", "departure", "departures"}:
        return "Direction must be 'arrivals' or 'departures'."
    key = "arr_iata" if normalized_direction.startswith("arr") else "dep_iata"
    airport = airport_iata.strip().upper()
    try:
        data = _aviation("flights", {key: airport, "limit": max(1, min(limit, 100))})
    except RuntimeError as exc:
        return f"Error: {exc}"
    flights = data.get("data") or []
    if not flights:
        return f"No {normalized_direction} found for {airport}."
    return f"{normalized_direction.title()} at {airport}:\n\n" + ("\n" + "-" * 40 + "\n").join(
        _format_flight(flight) for flight in flights
    )


@tool
def get_airline_info(search: str) -> str:
    """Look up airline details by name or code (requires Aviationstack API key)."""
    try:
        data = _aviation("airlines", {"search": search.strip(), "limit": 5})
    except RuntimeError as exc:
        return f"Error: {exc}"
    airlines = data.get("data") or []
    if not airlines:
        return f"No airlines found for '{search}'."
    return "\n\n".join(_format_airline(airline) for airline in airlines)


@tool
def get_weather_forecast(city: str, days: int = 7) -> str:
    """Get a daily weather forecast for a city, for up to 16 days."""
    if not 1 <= days <= 16:
        return "Forecast days must be between 1 and 16."
    try:
        geocoding = _get_json(
            OPEN_METEO_GEOCODING_URL,
            {"name": city.split(",")[0].strip(), "count": 1, "language": "en", "format": "json"},
        )
        places = geocoding.get("results") or []
        if not places:
            return f"Could not find a location called '{city}'."
        place = places[0]
        forecast = _get_json(
            OPEN_METEO_FORECAST_URL,
            {
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum",
                "timezone": "auto",
                "forecast_days": days,
            },
        )
    except (RuntimeError, KeyError, TypeError) as exc:
        return f"Error: {exc}"

    daily = forecast.get("daily") or {}
    lines = [f"{days}-day forecast for {place['name']}, {place.get('country', '')}:"]
    try:
        for index, day in enumerate(daily.get("time") or []):
            lines.append(
                f"  {day}: {WMO_CODES.get(daily['weather_code'][index], 'N/A')}, "
                f"{daily['temperature_2m_min'][index]} to {daily['temperature_2m_max'][index]} C, "
                f"rain {daily['precipitation_sum'][index]} mm"
            )
    except (KeyError, IndexError, TypeError) as exc:
        return f"Error: the weather service returned an incomplete forecast ({exc})."
    return "\n".join(lines)


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert an amount between currencies using current exchange rates."""
    source = from_currency.strip().upper()
    target = to_currency.strip().upper()
    try:
        data = _get_json(f"{EXCHANGE_RATES_URL}/{source}")
    except RuntimeError as exc:
        return f"Error: {exc}"
    if data.get("result") != "success":
        return f"Could not fetch rates for {source}."
    rate = (data.get("rates") or {}).get(target)
    if rate is None:
        return f"Unknown currency code: {target}"
    return (
        f"{amount:,.2f} {source} = {amount * rate:,.2f} {target} "
        f"(1 {source} = {rate:.4f} {target}, updated "
        f"{data.get('time_last_update_utc', 'recently')})"
    )


@tool
def estimate_trip_budget(
    days: int,
    travelers: int = 1,
    flights_total: float = 0,
    hotel_per_night: float = 0,
    food_per_person_per_day: float = 0,
    transport_per_person_per_day: float = 0,
    activities_per_person_per_day: float = 0,
    other_total: float = 0,
    currency: str = "USD",
) -> str:
    """Calculate trip costs; flight/other totals cover all travelers and hotel is per night."""
    if days < 1 or travelers < 1:
        return "Days and travelers must both be at least 1."
    costs = (
        flights_total,
        hotel_per_night,
        food_per_person_per_day,
        transport_per_person_per_day,
        activities_per_person_per_day,
        other_total,
    )
    if any(not math.isfinite(cost) or cost < 0 for cost in costs):
        return "Trip costs must be finite, non-negative amounts."
    nights = days - 1
    hotel = hotel_per_night * nights
    food = food_per_person_per_day * days * travelers
    transport = transport_per_person_per_day * days * travelers
    activities = activities_per_person_per_day * days * travelers
    total = flights_total + hotel + food + transport + activities + other_total
    return (
        f"Budget for {travelers} traveler(s), {days} days / {nights} nights ({currency}):\n"
        f"  Flights:    {flights_total:,.2f}\n"
        f"  Hotel:      {hotel:,.2f}\n"
        f"  Food:       {food:,.2f}\n"
        f"  Transport:  {transport:,.2f}\n"
        f"  Activities: {activities:,.2f}\n"
        f"  Other:      {other_total:,.2f}\n"
        f"  TOTAL:      {total:,.2f}  (about {total / travelers:,.2f} per person)"
    )


web_search = None
if os.getenv("TAVILY_API_KEY"):
    @tool
    def web_search(query: str) -> str:
        """Search the web for current information and return results with sources."""
        try:
            from tavily import TavilyClient

            result = TavilyClient(api_key=os.environ["TAVILY_API_KEY"]).search(
                query=query,
                max_results=5,
            )
            results = result.get("results") or []
            if not results:
                return "Web search returned no results."
            return "\n\n".join(
                f"{item.get('title', 'Result')}\n{item.get('url', '')}\n{item.get('content', '')}"
                for item in results
            )
        except Exception as exc:
            return f"Web search failed: {exc}"


GENERAL_TOOLS = [
    calculator,
    get_today,
    get_weather,
    browser_read,
    search_uploaded_document,
    remember_this,
    recall_memory,
]
if web_search is not None:
    GENERAL_TOOLS.append(web_search)

TRAVEL_TOOLS = [
    calculator,
    get_today,
    get_weather,
    get_country_info,
    get_main_airports,
    search_airports,
    get_flight_status,
    search_flights_by_route,
    list_airlines_on_route,
    get_airport_board,
    get_airline_info,
    get_weather_forecast,
    convert_currency,
    estimate_trip_budget,
]
