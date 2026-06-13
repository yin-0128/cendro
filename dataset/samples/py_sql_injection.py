import sqlite3


def find_users_by_name(conn: sqlite3.Connection, name: str):
    cursor = conn.cursor()
    query = "SELECT id, email FROM users WHERE name = '" + name + "'"
    cursor.execute(query)
    return cursor.fetchall()


def delete_user(conn: sqlite3.Connection, user_id):
    conn.execute(f"DELETE FROM users WHERE id = {user_id}")
    conn.commit()
