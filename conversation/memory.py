

import sqlite3
from pathlib import Path
from threading import Lock


BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "conversation_memory.db"


class ConversationMemory:
    def __init__(self, max_turns=6, db_path=DB_PATH):
        self.max_turns = max_turns
        self.db_path = Path(db_path)
        self.lock = Lock()

        self.db_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.initialize_database()

    def connect(self):
        connection = sqlite3.connect(
            str(self.db_path),
            timeout=30,
        )

        connection.row_factory = sqlite3.Row
        connection.execute(
            "PRAGMA busy_timeout = 30000"
        )

        return connection

    def initialize_database(self):
        with self.connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    collection TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            connection.execute("""
                CREATE INDEX IF NOT EXISTS
                idx_conversation_messages
                ON messages (
                    session_id,
                    collection,
                    id
                )
            """)

            connection.commit()

    # --------------------------------------------------
    # Retrieve Conversation History
    # --------------------------------------------------

    def get_history(self, session_id, collection):
        with self.lock:
            with self.connect() as connection:
                rows = connection.execute(
                    """
                    SELECT role, content
                    FROM (
                        SELECT id, role, content
                        FROM messages
                        WHERE session_id = ?
                          AND collection = ?
                        ORDER BY id DESC
                        LIMIT ?
                    )
                    ORDER BY id ASC
                    """,
                    (
                        session_id,
                        collection,
                        self.max_turns * 2,
                    ),
                ).fetchall()

        return [
            {
                "role": row["role"],
                "content": row["content"],
            }
            for row in rows
        ]

    # --------------------------------------------------
    # Save Conversation Turn
    # --------------------------------------------------

    def add_turn(
        self,
        session_id,
        collection,
        question,
        answer,
    ):
        with self.lock:
            with self.connect() as connection:
                connection.executemany(
                    """
                    INSERT INTO messages (
                        session_id,
                        collection,
                        role,
                        content
                    )
                    VALUES (?, ?, ?, ?)
                    """,
                    [
                        (
                            session_id,
                            collection,
                            "user",
                            question,
                        ),
                        (
                            session_id,
                            collection,
                            "assistant",
                            answer,
                        ),
                    ],
                )

                connection.commit()

    # --------------------------------------------------
    # Clear One Conversation Session
    # --------------------------------------------------

    def clear(self, session_id, collection):
        with self.lock:
            with self.connect() as connection:
                connection.execute(
                    """
                    DELETE FROM messages
                    WHERE session_id = ?
                      AND collection = ?
                    """,
                    (
                        session_id,
                        collection,
                    ),
                )

                connection.commit()

    # --------------------------------------------------
    # Clear All Conversations for a Collection
    # --------------------------------------------------

    def clear_collection(self, collection):
        """
        Delete all conversation history associated
        with a specific document collection,
        across every session.
        """

        with self.lock:
            with self.connect() as connection:
                cursor = connection.execute(
                    """
                    DELETE FROM messages
                    WHERE collection = ?
                    """,
                    (collection,),
                )

                deleted_messages = cursor.rowcount

                connection.commit()

        return deleted_messages


conversation_memory = ConversationMemory()
