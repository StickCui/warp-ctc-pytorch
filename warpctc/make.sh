#!/usr/bin/env bash
set -e

CURDIR=$(cd "$(dirname "$0")";pwd)
echo ${CURDIR}

echo "Make core shared object of libwarp_ctc.so"

cd ${CURDIR}/core

if [ ! -d "build" ]; then
  mkdir build
fi

cd build
cmake ..
make -j8

echo "Make extension of warp_ctc"
cd ${CURDIR}
python build.py

python3 setup.py build_ext --inplace
