import time

_cache = {}


def fetch_price(symbol, fetcher):
    """Return a cached price, refreshing only every 60 seconds."""
    if symbol in _cache:
        value, ts = _cache[symbol]
        if time.time() - ts < 60:
            return value
    value = fetcher(symbol)
    _cache[symbol] = (value, time.time())
    return value


def add_listener(event, listeners=[]):
    listeners.append(event)
    return listeners
