#!/usr/bin/env bash
set -euo pipefail

build_root="$(mktemp -d)"
trap 'rm -rf "$build_root"' EXIT

if [[ "$(uname -s)" == Darwin ]]; then
    brew install blitz boost pkg-config
    prefix="$(brew --prefix)"
    export CC=clang CXX=clang++
    export PKG_CONFIG_PATH="$prefix/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
else
    dnf install -y boost-devel git pkgconf-pkg-config
    prefix=/usr/local
    export PKG_CONFIG_PATH="$prefix/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"

    git clone --depth 1 --branch 1.0.2 https://github.com/blitzpp/blitz.git "$build_root/blitz"
    cmake -S "$build_root/blitz" -B "$build_root/blitz-build" \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$prefix" \
        -DCMAKE_INSTALL_LIBDIR=lib \
        -DBUILD_SHARED_LIBS=ON \
        -DBUILD_TESTING=OFF
    cmake --build "$build_root/blitz-build" --parallel 2
    cmake --install "$build_root/blitz-build"
    ldconfig "$prefix/lib"
fi

git init "$build_root/libmpdataxx"
git -C "$build_root/libmpdataxx" remote add origin https://github.com/igfuw/libmpdataxx.git
git -C "$build_root/libmpdataxx" fetch --depth 1 origin 8365cd2a475f51d2ce7706e9a836e9882c2cf2fe
git -C "$build_root/libmpdataxx" checkout --detach FETCH_HEAD

cmake -S "$build_root/libmpdataxx/libmpdata++" -B "$build_root/libmpdataxx-build" \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    -DCMAKE_POLICY_DEFAULT_CMP0074=NEW \
    -DCMAKE_POLICY_DEFAULT_CMP0167=OLD \
    -DCMAKE_PREFIX_PATH="$prefix" \
    -DCMAKE_INSTALL_PREFIX="$prefix"
cmake --install "$build_root/libmpdataxx-build"