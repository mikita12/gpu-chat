from fastapi import APIRouter, HTTPException, Request, Response
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.auth import (
    CurrentUser,
    DbDep,
    SettingsDep,
    create_session,
    delete_session,
    hash_password,
    validate_credentials,
    verify_password,
)
from app.models import User
from app.schemas import LoginRequest, RegisterRequest, UserOut

router = APIRouter(prefix="/api/auth")


def _set_session_cookie(response: Response, request: Request, settings: SettingsDep, token: str) -> None:
    response.set_cookie(
        key=settings.session_cookie_name,
        value=token,
        max_age=int(settings.session_ttl_seconds),
        httponly=True,
        samesite="lax",
        # Secure-aware, not hardcoded: a plain-HTTP LAN deployment (the
        # default here) still needs the cookie to be set at all, which a
        # hardcoded Secure=True would silently prevent.
        secure=request.url.scheme == "https",
        path="/",
    )


@router.post("/register", response_model=UserOut, status_code=201)
async def register(
    req: RegisterRequest, request: Request, response: Response, db: DbDep, settings: SettingsDep
) -> UserOut:
    error = validate_credentials(req.username, req.password)
    if error is not None:
        raise HTTPException(status_code=400, detail={"message": error, "code": "invalid_request"})
    user = User(username=req.username, password_hash=hash_password(req.password))
    db.add(user)
    try:
        await db.flush()
    except IntegrityError as exc:
        await db.rollback()
        raise HTTPException(
            status_code=409, detail={"message": "username already taken", "code": "username_taken"}
        ) from exc
    await db.commit()
    # Register also signs the new account in - a second login round trip
    # would add nothing given there's no email verification step to wait on.
    token = await create_session(db, user, settings.session_ttl_seconds)
    _set_session_cookie(response, request, settings, token)
    return UserOut(username=user.username)


@router.post("/login", response_model=UserOut)
async def login(
    req: LoginRequest, request: Request, response: Response, db: DbDep, settings: SettingsDep
) -> UserOut:
    result = await db.execute(select(User).where(User.username == req.username))
    user = result.scalar_one_or_none()
    # Never reveal which of username/password was wrong.
    invalid = HTTPException(
        status_code=401, detail={"message": "invalid username or password", "code": "invalid_credentials"}
    )
    if user is None:
        raise invalid
    if not verify_password(user.password_hash, req.password):
        raise invalid
    token = await create_session(db, user, settings.session_ttl_seconds)
    _set_session_cookie(response, request, settings, token)
    return UserOut(username=user.username)


@router.post("/logout", status_code=204)
async def logout(request: Request, response: Response, db: DbDep, settings: SettingsDep) -> None:
    token = request.cookies.get(settings.session_cookie_name)
    if token is not None:
        await delete_session(db, token)
    response.delete_cookie(settings.session_cookie_name, path="/")


@router.get("/me", response_model=UserOut)
async def me(user: CurrentUser) -> UserOut:
    return UserOut(username=user.username)
