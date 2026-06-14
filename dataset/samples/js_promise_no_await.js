function saveUser(db, user) {
  // returns immediately; caller can't know if the write succeeded
  db.collection("users").insertOne(user);
  return { ok: true };
}

async function chargeAndEmail(payments, mailer, order) {
  payments.charge(order.amount);
  mailer.sendReceipt(order.email);
}

module.exports = { saveUser, chargeAndEmail };
