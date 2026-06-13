def order_summaries(db, customer_ids):
    summaries = []
    for cid in customer_ids:
        customer = db.query("SELECT name FROM customers WHERE id = ?", cid)
        orders = db.query("SELECT total FROM orders WHERE customer_id = ?", cid)
        summaries.append({"name": customer["name"], "total": sum(o["total"] for o in orders)})
    return summaries
