import time


def call_with_retry(fn, *args):
    while True:
        try:
            return fn(*args)
        except Exception:
            time.sleep(1)


def safe_int(value):
    try:
        return int(value)
    except Exception:
        return 0
