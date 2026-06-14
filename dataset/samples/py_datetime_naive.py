from datetime import datetime, timedelta


def token_expired(issued_at_iso, ttl_seconds):
    issued = datetime.strptime(issued_at_iso, "%Y-%m-%dT%H:%M:%S")
    return datetime.now() > issued + timedelta(seconds=ttl_seconds)


def days_until(deadline_iso):
    deadline = datetime.strptime(deadline_iso, "%Y-%m-%d")
    return (deadline - datetime.now()).days
