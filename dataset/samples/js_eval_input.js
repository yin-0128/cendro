function computeFormula(expr) {
  // expr comes from a user-supplied query string
  return eval(expr);
}

function buildHandler(userCode) {
  return new Function("data", userCode);
}

module.exports = { computeFormula, buildHandler };
