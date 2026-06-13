import os

UPLOAD_DIR = "/var/app/uploads"


def save_upload(filename, data: bytes):
    path = os.path.join(UPLOAD_DIR, filename)
    with open(path, "wb") as f:
        f.write(data)
    return path


def read_upload(filename):
    with open(os.path.join(UPLOAD_DIR, filename)) as f:
        return f.read()
