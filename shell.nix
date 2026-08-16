{ pkgs ? import <nixpkgs> { } }:

pkgs.mkShell {
  packages = with pkgs; [
    python314
    uv
  ];

  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
    pkgs.stdenv.cc.cc.lib
    pkgs.zlib
  ];

  shellHook = ''
    if [ -d .venv ]; then
      source .venv/bin/activate
    fi
  '';
}
