#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" == Darwin ]]; then
    brew install gcc
else
    dnf install -y gcc-gfortran
fi