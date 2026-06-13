import json


def load_settings(path):
    f = open(path)
    settings = json.loads(f.read())
    return settings


def write_lines(path, lines):
    f = open(path, "w")
    for line in lines:
        f.write(line + "\n")
    f.close()
