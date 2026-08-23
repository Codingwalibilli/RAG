import json
import sqlite3
from langchain_core.documents import Document
from config import DB_PATH

def get_connection():
    conn = sqlite3.connect(DB_PATH)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS parents (
            parent_id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            metadata TEXT
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS children (
            chunk_id TEXT PRIMARY KEY,
            parent_id TEXT NOT NULL,
            content TEXT NOT NULL,
            metadata TEXT,
            FOREIGN KEY (parent_id)
                REFERENCES parents(parent_id)
        )
    """)

    return conn

def store_parents(parents):
    conn = get_connection()

    for parent in parents:
        conn.execute(
            """
            INSERT OR REPLACE INTO parents
            (parent_id, content, metadata)
            VALUES (?, ?, ?)
            """,
            (
                parent.metadata["parent_id"],
                parent.page_content,
                json.dumps(parent.metadata),
            )
        )

    conn.commit()
    conn.close()

def store_children(children):
    conn = get_connection()

    for child in children:
        conn.execute(
            """
            INSERT OR REPLACE INTO children
            (chunk_id, parent_id, content, metadata)
            VALUES (?, ?, ?, ?)
            """,
            (
                child.metadata["chunk_id"],
                child.metadata["parent_id"],
                child.page_content,
                json.dumps(child.metadata),
            )
        )

    conn.commit()
    conn.close()

def get_parent(parent_id):
    conn = get_connection()

    row = conn.execute(
        """
        SELECT content, metadata
        FROM parents
        WHERE parent_id = ?
        """,
        (parent_id,)
    ).fetchone()

    conn.close()

    if row is None:
        return None

    return Document(
        page_content=row[0],
        metadata=json.loads(row[1]),
    )

def get_parents(parent_ids):
    if not parent_ids:
        return []

    conn = get_connection()

    placeholders = ",".join("?" for _ in parent_ids)

    rows = conn.execute(
        f"""
        SELECT content, metadata
        FROM parents
        WHERE parent_id IN ({placeholders})
        """,
        list(parent_ids)
    ).fetchall()

    conn.close()

    return [
        Document(
            page_content=content,
            metadata=json.loads(metadata),
        )
        for content, metadata in rows
    ]

def get_children():
    conn = get_connection()

    rows = conn.execute(
        """
        SELECT content, metadata
        FROM children
        """
    ).fetchall()

    conn.close()

    return [
        Document(
            page_content=content,
            metadata=json.loads(metadata),
        )
        for content, metadata in rows
    ]