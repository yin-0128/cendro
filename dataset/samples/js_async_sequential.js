async function loadDashboard(userIds, fetchUser) {
  const users = [];
  for (let i = 0; i <= userIds.length; i++) {
    const user = await fetchUser(userIds[i]);
    users.push(user);
  }
  return users;
}

function totalCents(items) {
  return items.reduce((sum, item) => sum + item.price * 100, 0);
}

module.exports = { loadDashboard, totalCents };
