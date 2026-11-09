import json
import sqlite3
from contextlib import closing
from pathlib import Path
from uuid import uuid4


def dialogue_prompt(turns, message):
    lines = [f"User: {t['user']}\nAssistant: {t['assistant']}" for t in turns]
    return "\n".join([*lines, f"User: {message}\nAssistant:"])


class SessionStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as db, db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute(
                """CREATE TABLE IF NOT EXISTS sessions (
                   id TEXT PRIMARY KEY, model TEXT NOT NULL, revision INTEGER NOT NULL,
                   turns TEXT NOT NULL)"""
            )

    def connect(self):
        return sqlite3.connect(self.path, timeout=10)

    def create(self, model):
        session = {"id": uuid4().hex, "model": model, "revision": 0, "turns": []}
        with closing(self.connect()) as db, db:
            db.execute(
                "INSERT INTO sessions VALUES (?, ?, 0, '[]')", (session["id"], model)
            )
        return session

    def get(self, session_id):
        with closing(self.connect()) as db:
            row = db.execute(
                "SELECT model, revision, turns FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            raise KeyError(session_id)
        model, revision, turns = row
        return {
            "id": session_id,
            "model": model,
            "revision": revision,
            "turns": json.loads(turns),
        }

    def replace(self, session, turns):
        with closing(self.connect()) as db, db:
            cursor = db.execute(
                "UPDATE sessions SET turns = ?, revision = revision + 1 WHERE id = ? AND revision = ?",
                (
                    json.dumps(turns, ensure_ascii=False),
                    session["id"],
                    session["revision"],
                ),
            )
            if cursor.rowcount != 1:
                return None
        return {**session, "revision": session["revision"] + 1, "turns": turns}
