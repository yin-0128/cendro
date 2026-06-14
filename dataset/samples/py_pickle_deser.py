import pickle


def load_session(blob: bytes):
    # Restore a user session object from a cookie payload.
    return pickle.loads(blob)


def cache_set(store, key, value):
    store[key] = pickle.dumps(value)
