import random
import string


def make_reset_token(length=16):
    chars = string.ascii_letters + string.digits
    return "".join(random.choice(chars) for _ in range(length))


def make_session_id():
    return str(random.randint(100000, 999999))
