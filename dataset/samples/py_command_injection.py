import os
import subprocess


def ping(host):
    os.system("ping -c 1 " + host)


def archive(folder):
    subprocess.call(f"tar -czf backup.tgz {folder}", shell=True)
