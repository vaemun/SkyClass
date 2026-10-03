import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "chat_history.db")


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    conn = get_conn()
    c = conn.cursor()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS chat (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL DEFAULT 'New Chat',
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            type TEXT NOT NULL CHECK(type IN ('document','image','link')),
            chat_id INTEGER NOT NULL REFERENCES chat(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id INTEGER NOT NULL REFERENCES chat(id) ON DELETE CASCADE,
            sender TEXT NOT NULL CHECK(sender IN ('user','ai')),
            content TEXT NOT NULL,
            timestamp TEXT DEFAULT (datetime('now'))
        );
    """)
    conn.commit()
    conn.close()


def create_chat(title="New Chat"):
    conn = get_conn()
    c = conn.cursor()
    c.execute("INSERT INTO chat (title) VALUES (?)", (title,))
    chat_id = c.lastrowid
    conn.commit()
    conn.close()
    return chat_id


def get_chats():
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT id, title, created_at FROM chat ORDER BY created_at DESC")
    rows = c.fetchall()
    conn.close()
    return rows


def get_chat(chat_id):
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT id, title, created_at FROM chat WHERE id=?", (chat_id,))
    row = c.fetchone()
    conn.close()
    return row


def delete_chat(chat_id):
    conn = get_conn()
    c = conn.cursor()
    c.execute("DELETE FROM messages WHERE chat_id=?", (chat_id,))
    c.execute("DELETE FROM sources WHERE chat_id=?", (chat_id,))
    c.execute("DELETE FROM chat WHERE id=?", (chat_id,))
    conn.commit()
    conn.close()


def update_chat_title(chat_id, title):
    conn = get_conn()
    c = conn.cursor()
    c.execute("UPDATE chat SET title=? WHERE id=?", (title, chat_id))
    conn.commit()
    conn.close()


def add_message(chat_id, sender, content):
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "INSERT INTO messages (chat_id, sender, content) VALUES (?, ?, ?)",
        (chat_id, sender, content),
    )
    msg_id = c.lastrowid
    conn.commit()
    conn.close()
    return msg_id


def get_messages(chat_id):
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "SELECT id, sender, content, timestamp FROM messages WHERE chat_id=? ORDER BY timestamp",
        (chat_id,),
    )
    rows = c.fetchall()
    conn.close()
    return rows


def add_source(chat_id, name, type_):
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "INSERT INTO sources (name, type, chat_id) VALUES (?, ?, ?)",
        (name, type_, chat_id),
    )
    src_id = c.lastrowid
    conn.commit()
    conn.close()
    return src_id


def get_sources(chat_id):
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "SELECT id, name, type FROM sources WHERE chat_id=? ORDER BY id",
        (chat_id,),
    )
    rows = c.fetchall()
    conn.close()
    return rows


def delete_source(source_id):
    conn = get_conn()
    c = conn.cursor()
    c.execute("DELETE FROM sources WHERE id=?", (source_id,))
    conn.commit()
    conn.close()
