"""
Long-term memory store using Supabase PostgreSQL + pgvector.

Implements semantic storage and retrieval of memory facts.
"""

from loguru import logger
import uuid
from datetime import datetime
from typing import List, Iterable
from sqlalchemy import select, update, and_, text
from memory.schemas import MemoryFact, ReminderIntent
from memory.policies import apply_decay, should_prune
from infrastructure.db.sql_client import mem_facts_table, get_session

def _to_datetime(timestamp: float | None) -> datetime | None:
    """Convert Unix timestamp to datetime for PostgreSQL."""
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp)

