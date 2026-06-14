def is_root(user_id):
    return user_id is 0


def status_label(code):
    if code is 200:
        return "ok"
    if code is 404:
        return "not found"
    return "unknown"
