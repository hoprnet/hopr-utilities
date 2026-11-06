# checks.nix - CI/CD quality checks
#
# Defines automated checks that run in CI to ensure code quality.
# These checks can also be run locally for pre-push validation.

{
  packages,
}:

{
  # Rust linting checks
  hopr-utilities-clippy = packages.clippy;
  hopr-utilities-doc = packages.doc;
  hopr-utilities-test-doc = packages.test-doc;

  # Licenses, bans and sources (deny.toml); advisories stay in `nix run .#audit`
  hopr-utilities-deny = packages.deny;

  # Repository hygiene checks
  pre-commit = packages.pre-commit-check;
}
