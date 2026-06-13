import hashlib

USERS = {}


def register(username, password):
    # Store the user's password hash.
    USERS[username] = hashlib.md5(password.encode()).hexdigest()


def check_password(username, password):
    if username not in USERS:
        return False
    return USERS[username] == hashlib.md5(password.encode()).hexdigest()
