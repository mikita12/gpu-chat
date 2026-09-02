import datetime
import hashlib
import secrets
from collections.abc import AsyncIterator
from typing import Annotated

from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from fastapi import Depends, HTTPException, Request
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.config import Settings, get_settings
from app.db import get_db_session
from app.models import Session as SessionRow
from app.models import User, utcnow

MIN_USERNAME_LENGTH = 3
MAX_USERNAME_LENGTH = 32
MIN_PASSWORD_LENGTH = 8

_hasher = PasswordHasher()


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    try:
        return _hasher.verify(password_hash, password)
    except VerifyMismatchError:
        return False


def _hash_token(token: str) -> str:
    # The cookie carries the raw token; only its hash is ever stored, so a
    # copy of the database alone can't be replayed as a live session.
    return hashlib.sha256(token.encode()).hexdigest()


def validate_credentials(username: str, password: str) -> str | None:
    """Returns an error message if the credentials fail basic shape rules,
    None if they're acceptable. Deliberately minimal - this is LAN-trust
    auth, not internet-facing hardening (see README)."""
    if not (MIN_USERNAME_LENGTH <= len(username) <= MAX_USERNAME_LENGTH):
        return f"username must be {MIN_USERNAME_LENGTH}-{MAX_USERNAME_LENGTH} characters"
    if len(password) < MIN_PASSWORD_LENGTH:
        return f"password must be at least {MIN_PASSWORD_LENGTH} characters"
    return None


async def create_session(db: AsyncSession, user: User, ttl_seconds: float) -> str:
    """Returns the raw token to send as a cookie - never stored itself,
    only its hash (see _hash_token)."""
    token = secrets.token_urlsafe(32)
    now = utcnow()
    db.add(
        SessionRow(
            user_id=user.id,
            token_hash=_hash_token(token),
            created_at=now,
            expires_at=now + datetime.timedelta(seconds=ttl_seconds),
        )
    )
    await db.commit()
    return token


async def delete_session(db: AsyncSession, token: str) -> None:
    await db.execute(delete(SessionRow).where(SessionRow.token_hash == _hash_token(token)))
    await db.commit()


async def get_user_for_token(db: AsyncSession, token: str) -> User | None:
    """Looks up the user for a raw session token, lazily deleting it (and
    returning None) if it's expired - sessions live in SQLite, not a
    process dict, so nothing else ever reclaims a stale row on its own
    besides this lookup and the periodic background sweep in app/db.py."""
    now = utcnow()
    result = await db.execute(select(SessionRow).where(SessionRow.token_hash == _hash_token(token)))
    session_row = result.scalar_one_or_none()
    if session_row is None:
        return None
    if session_row.expires_at < now:
        await db.delete(session_row)
        await db.commit()
        return None
    user_result = await db.execute(select(User).where(User.id == session_row.user_id))
    return user_result.scalar_one_or_none()


async def get_db(request: Request) -> AsyncIterator[AsyncSession]:
    # A FastAPI generator dependency, not a plain return - `return db` after
    # a single `async for ... : return db` step would leave
    # get_db_session()'s `async with session_factory() as db:` block
    # suspended forever (nothing ever resumes the generator to let it
    # exit), which never actually closes the session/connection - it was
    # only ever reclaimed later by garbage collection. Declaring this
    # function itself as the yielding dependency lets FastAPI close it
    # properly once the request (not the response body - see
    # app/conversations_api.py's _stream_and_persist for the streaming
    # case) is done.
    factory: async_sessionmaker[AsyncSession] = request.app.state.db_session_factory
    async for db in get_db_session(factory):
        yield db


DbDep = Annotated[AsyncSession, Depends(get_db)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


async def require_session(request: Request, db: DbDep, settings: SettingsDep) -> User:
    """FastAPI dependency for every session-gated route. Reads the cookie
    by name directly from the request (rather than via a `Cookie()`
    parameter default, which can't be bound to a runtime Settings value)
    so the cookie name stays configurable through settings.session_cookie_name."""
    token = request.cookies.get(settings.session_cookie_name)
    if token is None:
        raise HTTPException(status_code=401, detail={"message": "not signed in", "code": "unauthorized"})
    user = await get_user_for_token(db, token)
    if user is None:
        raise HTTPException(status_code=401, detail={"message": "session expired or invalid", "code": "unauthorized"})
    return user


CurrentUser = Annotated[User, Depends(require_session)]
