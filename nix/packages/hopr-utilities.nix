# hopr-utilities.nix - hopr-utilities library package definitions
#
# Defines all variants of the hopr-utilities library for different platforms.
# hopr-utilities is a Rust library crate of shared utility functions for HOPR.

{
  lib,
  builders,
  sources,
  hoprUtilitiesCrateInfo,
  rev,
  nixLib,
}:

let
  # Common build arguments for hopr-utilities variants
  mkHoprUtilitiesBuildArgs =
    { src, depsSrc }:
    {
      inherit src depsSrc rev;
      cargoToml = ./../../Cargo.toml;
    };

  localArgs = mkHoprUtilitiesBuildArgs {
    src = sources.main;
    depsSrc = sources.deps;
  };

  mkHoprUtilitiesPlatformPackages =
    platform:
    let
      name = "lib-hopr-utilities-${platform}";
    in
    {
      "${name}" = builders.${platform}.callPackage nixLib.mkRustLibrary localArgs;
    }
    // lib.optionalAttrs (lib.hasSuffix "-linux" platform) {
      "${name}-dev" = builders.${platform}.callPackage nixLib.mkRustLibrary (
        localArgs // { CARGO_PROFILE = "dev"; }
      );
    };

  hoprUtilitiesPlatformPackages = builtins.foldl' (a: b: a // b) { } (
    map mkHoprUtilitiesPlatformPackages [
      "x86_64-linux"
      "aarch64-linux"
      "x86_64-darwin"
      "aarch64-darwin"
    ]
  );
in
{
  lib-hopr-utilities = builders.local.callPackage nixLib.mkRustLibrary localArgs;

  clippy = builders.local.callPackage nixLib.mkRustLibrary (
    localArgs
    // {
      runClippy = true;
    }
  );
}
// hoprUtilitiesPlatformPackages
