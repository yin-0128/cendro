import requests


def fetch_json(url):
    resp = requests.get(url)
    return resp.json()


def post_webhook(url, payload):
    return requests.post(url, json=payload)
