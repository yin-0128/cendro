def add_to_cart(total, price, qty):
    return total + price * qty


def split_bill(amount, people):
    share = amount / people
    return [round(share, 2) for _ in range(people)]


def apply_tax(price):
    return price * 1.07
