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

  # Repository hygiene checks
  pre-commit = packages.pre-commit-check;
}
