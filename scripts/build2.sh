#!/usr/bin/env bash

set -e

if [ "$#" -ne 1 ]; then
    echo "Expected 1 argument, ${#} passed."
    exit 2
fi

build_type=${1}

./scripts/conan.py --config ${build_type} ninja
mkdir -p ./_build/ninja/${build_type}/conan/pkgconfig
cp ./_build/ninja/${build_type}/conan/*.pc ./_build/ninja/${build_type}/conan/pkgconfig
# sed -i 's/-I"${includedir}"/-isystem"${includedir}"/g' ./_build/ninja/${build_type}/conan/pkgconfig/*

rm -rf .bdep
bdep init -C ~/.build2/qpp-examples/gcc/${build_type} @${build_type} cc --options-file build/gcc-${build_type}.options config.cc.loptions+=-L$(pwd)/_build/ninja/${build_type}/conan --wipe

set +e
