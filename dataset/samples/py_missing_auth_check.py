def delete_account(request, db):
    user_id = request.args.get("user_id")
    db.execute("DELETE FROM accounts WHERE id = ?", (user_id,))
    return {"deleted": user_id}


def get_invoice(request, db):
    invoice_id = request.args.get("id")
    return db.query("SELECT * FROM invoices WHERE id = ?", (invoice_id,))
