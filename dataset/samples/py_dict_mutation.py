def drop_inactive(users):
    for uid, user in users.items():
        if not user.get("active"):
            del users[uid]
    return users


def increment_counts(counts, items):
    for item in items:
        counts[item] += 1
    return counts
