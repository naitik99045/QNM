"""
SQLite Persistent Storage for QNM Concept Memory and Telemetry.
Stores multi-turn sessions, turns, symbol tables, discrete code arrays, and coarse gists.
"""

import sqlite3
import uuid
import json
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator
from qnm.guard.symbol_table import SymbolTable


class MemoryStore:
    """Manages the SQLite database for session turns and compression metrics."""

    def __init__(self, db_path: str = "artifacts/concept_memory.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    title TEXT
                )
            """)
            # Turns table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS turns (
                    turn_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    raw_prompt TEXT NOT NULL,
                    masked_prompt TEXT NOT NULL,
                    symbols_json TEXT NOT NULL,
                    coarse_gist TEXT NOT NULL,
                    discrete_codes TEXT NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                )
            """)
            # Telemetry logs
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS telemetry_logs (
                    log_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    raw_tokens INTEGER NOT NULL,
                    compressed_tokens INTEGER NOT NULL,
                    skeleton_tokens INTEGER NOT NULL,
                    final_tokens INTEGER NOT NULL,
                    latency_ms REAL NOT NULL
                )
            """)
            conn.commit()

    def ensure_session(self, session_id: str, title: Optional[str] = None):
        """Ensure a session record exists."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT session_id FROM sessions WHERE session_id = ?", (session_id,))
            if not cursor.fetchone():
                now = datetime.now(timezone.utc).isoformat()
                cursor.execute(
                    "INSERT INTO sessions (session_id, created_at, title) VALUES (?, ?, ?)",
                    (session_id, now, title or f"Session {session_id[:8]}")
                )
                conn.commit()

    def save_turn(
        self,
        session_id: str,
        role: str,
        raw_prompt: str,
        masked_prompt: str,
        symbol_table: SymbolTable,
        coarse_gist: str,
        discrete_codes: List[int],
    ) -> str:
        """Persist a conversation turn."""
        self.ensure_session(session_id)
        turn_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        symbols_json = symbol_table.to_json()
        codes_json = json.dumps(discrete_codes)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO turns (
                    turn_id, session_id, role, timestamp,
                    raw_prompt, masked_prompt, symbols_json,
                    coarse_gist, discrete_codes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    turn_id, session_id, role, now,
                    raw_prompt, masked_prompt, symbols_json,
                    coarse_gist, codes_json
                )
            )
            conn.commit()
        return turn_id

    def get_session_turns(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve historical turns for a given session."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM turns
                WHERE session_id = ?
                ORDER BY timestamp ASC
                LIMIT ?
                """,
                (session_id, limit)
            )
            rows = cursor.fetchall()
            results = []
            for r in rows:
                results.append({
                    "turn_id": r["turn_id"],
                    "session_id": r["session_id"],
                    "role": r["role"],
                    "timestamp": r["timestamp"],
                    "raw_prompt": r["raw_prompt"],
                    "masked_prompt": r["masked_prompt"],
                    "symbols": json.loads(r["symbols_json"]),
                    "coarse_gist": r["coarse_gist"],
                    "discrete_codes": json.loads(r["discrete_codes"])
                })
            return results

    def clear_session(self, session_id: Optional[str] = None):
        """Clear turns for a specific session or all sessions if None."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            if session_id:
                cursor.execute("DELETE FROM turns WHERE session_id = ?", (session_id,))
                cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            else:
                cursor.execute("DELETE FROM turns")
                cursor.execute("DELETE FROM sessions")
            conn.commit()

    def record_telemetry(
        self,
        session_id: str,
        raw_tokens: int,
        compressed_tokens: int,
        skeleton_tokens: int,
        final_tokens: int,
        latency_ms: float
    ):
        """Save telemetry statistics for an inference run."""
        log_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO telemetry_logs (
                    log_id, session_id, timestamp,
                    raw_tokens, compressed_tokens, skeleton_tokens,
                    final_tokens, latency_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (log_id, session_id, now, raw_tokens, compressed_tokens, skeleton_tokens, final_tokens, latency_ms)
            )
            conn.commit()

    def get_telemetry_stats(self) -> Dict[str, Any]:
        """Aggregate cumulative token savings and average latency."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT
                    COUNT(*) as total_requests,
                    SUM(raw_tokens) as total_raw,
                    SUM(compressed_tokens) as total_compressed,
                    SUM(skeleton_tokens) as total_skeleton,
                    SUM(final_tokens) as total_final,
                    AVG(latency_ms) as avg_latency_ms
                FROM telemetry_logs
            """)
            row = cursor.fetchone()
            if not row or row["total_requests"] == 0:
                return {
                    "total_requests": 0,
                    "total_raw_tokens": 0,
                    "total_compressed_tokens": 0,
                    "total_skeleton_tokens": 0,
                    "total_final_tokens": 0,
                    "context_saved_tokens": 0,
                    "context_saved_percent": 0.0,
                    "reasoning_saved_tokens": 0,
                    "reasoning_saved_percent": 0.0,
                    "avg_latency_ms": 0.0
                }

            total_raw = row["total_raw"] or 0
            total_compressed = row["total_compressed"] or 0
            total_skeleton = row["total_skeleton"] or 0
            total_final = row["total_final"] or 0
            context_saved = max(0, total_raw - total_compressed)
            context_saved_pct = (context_saved / total_raw * 100.0) if total_raw > 0 else 0.0
            reasoning_saved = max(0, total_final - total_skeleton)
            reasoning_saved_pct = (reasoning_saved / total_final * 100.0) if total_final > 0 else 0.0

            return {
                "total_requests": row["total_requests"],
                "total_raw_tokens": total_raw,
                "total_compressed_tokens": total_compressed,
                "total_skeleton_tokens": total_skeleton,
                "total_final_tokens": total_final,
                "context_saved_tokens": context_saved,
                "context_saved_percent": round(context_saved_pct, 1),
                "reasoning_saved_tokens": reasoning_saved,
                "reasoning_saved_percent": round(reasoning_saved_pct, 1),
                "avg_latency_ms": round(row["avg_latency_ms"] or 0.0, 2)
            }
