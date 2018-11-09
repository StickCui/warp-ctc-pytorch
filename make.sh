#!/usr/bin/env bash
set -e

TYPE="build"
if [ ! -n "$1" ]; then
  TYPE="build"
else
  TYPE=$1
fi

# Check args
case ${TYPE} in
  build)
    ;;
  install)
    ;;
  core)
    ;;
  *)
    echo "The input parameter incorrect."
    echo "The following parameters are allowed:"
    echo "  install: install the module in your user python path"
    echo "  build: build the module inplace"
    echo "  core: only build the Baidu Warp-CTC runtime library"
    exit 1
    ;;
esac

CURDIR=$(cd "$(dirname "$0")";pwd)
echo ${CURDIR}

echo "Make core shared object of libwarp_ctc.so"

cd ${CURDIR}/warpctc/core

if [ ! -d "build" ]; then
  mkdir build
fi

cd build
cmake ..
make -j8

case ${TYPE} in
  build)
    echo "Make extension of warp_ctc"
    cd ${CURDIR}

    python3 setup.py build_ext --inplace

    rm -rf build
    ;;
  install)
    echo "Install extension of warp_ctc"
    cd ${CURDIR}

    python3 setup.py install --user

    rm -rf build dist warpctc.egg-info
    ;;
  *)
    ;;
esac


