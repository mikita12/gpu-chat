import os

from alembic.config import Config as AlembicConfig
from sqlalchemy import create_engine, inspect

from alembic import command

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_alembic_upgrade_head_matches_the_orm_models(tmp_path: object) -> None:
    """Proves the checked-in migration actually creates the schema the ORM
    models describe - not just that some migration file exists. Runs the
    real `alembic upgrade head` (not Base.metadata.create_all, which every
    other test's session_factory fixture uses for speed) against a fresh
    temp database file."""
    db_path = os.path.join(str(tmp_path), "migration_test.db")

    alembic_cfg = AlembicConfig(os.path.join(REPO_ROOT, "alembic.ini"))
    alembic_cfg.set_main_option("script_location", os.path.join(REPO_ROOT, "alembic"))
    # Set directly rather than via the DATABASE_URL env var + get_settings():
    # get_settings() is @lru_cache'd process-wide, so an earlier test in the
    # same pytest run may have already cached a Settings instance built
    # before this env var existed - see alembic/env.py's _sync_database_url().
    alembic_cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    command.upgrade(alembic_cfg, "head")

    engine = create_engine(f"sqlite:///{db_path}")
    tables = set(inspect(engine).get_table_names())
    assert {"users", "sessions", "conversations", "messages"} <= tables
