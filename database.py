# database.py

from datetime import datetime

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    DateTime,
)

from sqlalchemy.orm import (
    declarative_base,
    sessionmaker,
)



# ============================================================
# DATABASE CONFIGURATION
# ============================================================

DATABASE_URL = "sqlite:///database_chatbot.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)

Base = declarative_base()


# ============================================================
# CONVERSATIONS TABLE
# ============================================================

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    thread_id = Column(
        String,
        unique=True,
        index=True,
        nullable=False,
    )

    user_id = Column(
        String,
        index=True,
        nullable=True,
    )

    title = Column(
        String,
        nullable=True,
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
    )

    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )


# ============================================================
# CHAT MESSAGES TABLE
# ============================================================

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    thread_id = Column(
        String,
        index=True,
        nullable=False,
    )

    role = Column(
        String,
        nullable=False,
    )

    content = Column(
        Text,
        nullable=False,
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
    )


# ============================================================
# LONG-TERM MEMORY TABLE
# ============================================================

class LongTermMemory(Base):
    __tablename__ = "long_term_memory"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    user_id = Column(
        String,
        index=True,
        nullable=False,
    )

    memory = Column(
        Text,
        nullable=False,
    )

    memory_type = Column(
        String,
        nullable=True,
    )

    created_at = Column(
        DateTime,
        default=datetime.utcnow,
    )

    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )


# ============================================================
# INITIALIZE DATABASE
# ============================================================

def init_db():
    """
    Create all database tables if they do not already exist.
    """

    Base.metadata.create_all(bind=engine)


# ============================================================
# CREATE OR UPDATE CONVERSATION
# ============================================================

def create_or_update_conversation(
    db,
    thread_id,
    user_id=None,
    title=None,
):
    """
    Create a conversation if it doesn't exist.

    If the conversation already exists, update its
    updated_at timestamp and optionally its title/user_id.
    """

    conversation = (
        db.query(Conversation)
        .filter(
            Conversation.thread_id == thread_id
        )
        .first()
    )

    if conversation is None:

        conversation = Conversation(
            thread_id=thread_id,
            user_id=user_id,
            title=title,
        )

        db.add(conversation)

    else:

        conversation.updated_at = datetime.utcnow()

        if title is not None:
            conversation.title = title

        if user_id is not None:
            conversation.user_id = user_id

    db.commit()
    db.refresh(conversation)

    return conversation


# ============================================================
# SAVE CHAT MESSAGE
# ============================================================

def save_chat_message(
    db,
    thread_id,
    role,
    content,
):
    """
    Save a single user/assistant/tool message.
    """

    message = ChatMessage(
        thread_id=thread_id,
        role=role,
        content=content,
    )

    db.add(message)

    # Update conversation timestamp
    conversation = (
        db.query(Conversation)
        .filter(
            Conversation.thread_id == thread_id
        )
        .first()
    )

    if conversation:
        conversation.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(message)

    return message


# ============================================================
# GET CHAT HISTORY
# ============================================================

def get_chat_history(
    db,
    thread_id,
):
    """
    Return all messages belonging to a thread.
    """

    messages = (
        db.query(ChatMessage)
        .filter(
            ChatMessage.thread_id == thread_id
        )
        .order_by(
            ChatMessage.id.asc()
        )
        .all()
    )

    return messages


def get_user_conversations(
    db,
    user_id,
):
    """Return conversations belonging to one browser session, newest first."""

    return (
        db.query(Conversation)
        .filter(Conversation.user_id == user_id)
        .order_by(Conversation.updated_at.desc())
        .all()
    )


def get_user_conversation(
    db,
    thread_id,
    user_id,
):
    """Return a conversation only when it belongs to the given browser session."""

    return (
        db.query(Conversation)
        .filter(
            Conversation.thread_id == thread_id,
            Conversation.user_id == user_id,
        )
        .first()
    )


def delete_user_conversation(
    db,
    thread_id,
    user_id,
):
    """Delete one conversation and its messages (only if it belongs to the session)."""

    conversation = get_user_conversation(db, thread_id, user_id)

    if conversation is None:
        return False

    db.query(ChatMessage).filter(
        ChatMessage.thread_id == thread_id
    ).delete(synchronize_session=False)

    db.delete(conversation)
    db.commit()

    return True


def delete_user_conversations(
    db,
    user_id,
):
    """Delete every conversation (and messages) belonging to one browser session."""

    thread_ids = [
        c.thread_id
        for c in db.query(Conversation.thread_id)
        .filter(Conversation.user_id == user_id)
        .all()
    ]

    if thread_ids:
        db.query(ChatMessage).filter(
            ChatMessage.thread_id.in_(thread_ids)
        ).delete(synchronize_session=False)

        db.query(Conversation).filter(
            Conversation.user_id == user_id
        ).delete(synchronize_session=False)

        db.commit()

    return len(thread_ids)


# ============================================================
# SAVE LONG-TERM MEMORY
# ============================================================

def save_memory(
    db,
    user_id,
    memory,
    memory_type=None,
):
    """
    Save a long-term memory for a user.
    """

    memory_item = LongTermMemory(
        user_id=user_id,
        memory=memory,
        memory_type=memory_type,
    )

    db.add(memory_item)
    db.commit()
    db.refresh(memory_item)

    return memory_item


# ============================================================
# GET LONG-TERM MEMORIES
# ============================================================

def get_memories(
    db,
    user_id,
):
    """
    Get all long-term memories for a user.
    """

    memories = (
        db.query(LongTermMemory)
        .filter(
            LongTermMemory.user_id == user_id
        )
        .order_by(
            LongTermMemory.created_at.asc()
        )
        .all()
    )

    return memories


# ============================================================
# DELETE MEMORY
# ============================================================

def delete_memory(
    db,
    memory_id,
    user_id,
):
    """
    Delete one memory belonging to a user.
    """

    memory = (
        db.query(LongTermMemory)
        .filter(
            LongTermMemory.id == memory_id,
            LongTermMemory.user_id == user_id,
        )
        .first()
    )

    if memory:
        db.delete(memory)
        db.commit()

        return True

    return False


# ============================================================
# DATABASE SESSION HELPER
# ============================================================

def get_db():
    """
    Create a database session.

    Usage:

        db = get_db()

        try:
            ...
        finally:
            db.close()
    """

    db = SessionLocal()

    try:
        yield db

    finally:
        db.close()


# ============================================================
# START DATABASE
# ============================================================

if __name__ == "__main__":

    init_db()

    print("Database initialized successfully.")