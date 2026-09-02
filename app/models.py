import datetime

from sqlalchemy import ForeignKey, Index
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def utcnow() -> datetime.datetime:
    # Naive on purpose: SQLite has no real timezone-aware storage, and
    # SQLAlchemy's DateTime reads values back naive regardless of what was
    # written - comparing a naive column value against an aware "now" (e.g.
    # datetime.now(UTC)) raises TypeError. Using naive UTC consistently,
    # everywhere a timestamp is created or compared, avoids that entirely.
    return datetime.datetime.now(datetime.UTC).replace(tzinfo=None)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(unique=True, index=True)
    password_hash: Mapped[str]
    created_at: Mapped[datetime.datetime] = mapped_column(default=utcnow)

    sessions: Mapped[list["Session"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    conversations: Mapped[list["Conversation"]] = relationship(
        back_populates="owner", cascade="all, delete-orphan"
    )


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    # SHA-256 hex digest of the opaque cookie token - never the raw token
    # itself, so a DB read alone can't be replayed as a live session.
    token_hash: Mapped[str] = mapped_column(unique=True, index=True)
    created_at: Mapped[datetime.datetime] = mapped_column(default=utcnow)
    expires_at: Mapped[datetime.datetime]

    user: Mapped[User] = relationship(back_populates="sessions")


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(primary_key=True)
    owner_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    title: Mapped[str | None] = mapped_column(default=None)
    model: Mapped[str]
    created_at: Mapped[datetime.datetime] = mapped_column(default=utcnow)
    updated_at: Mapped[datetime.datetime] = mapped_column(default=utcnow)

    owner: Mapped[User] = relationship(back_populates="conversations")
    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan", order_by="Message.id"
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("conversations.id"), index=True)
    role: Mapped[str]
    content: Mapped[str]
    # Reasoning-model "thinking" trace, stored for display only - never
    # rebuilt into the ChatMessage list sent back to Ollama on a later turn
    # (ChatMessage in app/schemas.py has no `thinking` field at all, so this
    # is structurally impossible to leak upstream by accident, not just a
    # rule to remember).
    thinking: Mapped[str | None] = mapped_column(default=None)
    created_at: Mapped[datetime.datetime] = mapped_column(default=utcnow)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


Index("ix_conversations_owner_updated", Conversation.owner_id, Conversation.updated_at.desc())
