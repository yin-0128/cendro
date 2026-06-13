import threading

counter = 0


def increment_many(times):
    global counter
    for _ in range(times):
        counter += 1


def run(workers, per_worker):
    threads = [threading.Thread(target=increment_many, args=(per_worker,)) for _ in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return counter
