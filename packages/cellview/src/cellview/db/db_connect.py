from collections.abc import Iterator
from contextlib import contextmanager
from typing import ParamSpec, TypeVar

import duckdb

from cellview.db.clean_up import clean_up_db
from cellview.db.db import CellViewDB

P = ParamSpec("P")  # For capturing arbitrary positional and keyword arguments
R = TypeVar("R")  # For preserving the return type


@contextmanager
def database_session(db: CellViewDB) -> Iterator[duckdb.DuckDBPyConnection]:
    """Context manager for database sessions.

    Handles connection setup and cleanup.
    """
    conn = db.connect()
    try:
        conn.begin()
        yield conn
        conn.commit()
    except Exception as e:
        clean_up_db(db, conn)
        conn.close()
        raise e
    finally:
        conn.close()
