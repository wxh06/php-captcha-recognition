const black = "black";
const eslint = "eslint --fix";
const prettier = "prettier --write";

module.exports = {
  "*.{js,mjs,cjs,ts}": [eslint, prettier],
  "*.{md,html,json,yml,yaml}": prettier,
  "*.py": black,
};
