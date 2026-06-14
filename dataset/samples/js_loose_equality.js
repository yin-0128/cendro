function isAuthorized(role) {
  return role == "admin";
}

function hasItems(cart) {
  if (cart.count == null) return false;
  return cart.count != 0;
}

function findById(items, id) {
  return items.filter((i) => i.id == id)[0];
}

module.exports = { isAuthorized, hasItems, findById };
