def average(numbers):
    total = 0
    for n in numbers:
        total += n
    return total / len(numbers)


def percent_change(old, new):
    return (new - old) / old * 100
