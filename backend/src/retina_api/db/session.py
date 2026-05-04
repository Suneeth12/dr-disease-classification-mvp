from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from retina_api.core.settings import settings
from retina_api.db.models import Base


connect_args = {}
if settings.database_url.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(settings.database_url, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def ensure_prediction_schema(target_engine=engine) -> None:
    if not settings.database_url.startswith("sqlite"):
        return
    with target_engine.begin() as connection:
        columns = {
            str(row[1])
            for row in connection.exec_driver_sql("PRAGMA table_info(predictions)").fetchall()
        }
        for column_name in ("ensemble_members", "ensemble_member_weights", "threshold_vector", "ereg_graph"):
            if column_name not in columns:
                connection.exec_driver_sql(f"ALTER TABLE predictions ADD COLUMN {column_name} TEXT")


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    ensure_prediction_schema(engine)
