import re


def extract_emails(documents):
    found = []
    for doc in documents:
        pattern = re.compile(r"[\w.]+@[\w.]+")
        found.extend(pattern.findall(doc))
    return found


def is_valid_id(s):
    return re.match("^[A-Z]{2}[0-9]{6}$", s) != None
