#!/bin/sh
python -m pip uninstall -y mmcv mmcv-lite
python -m pip install -U pip "setuptools>=70" wheel ninja psutil

export MMCV_WITH_OPS=1
export FORCE_CUDA=1
export MAX_JOBS=$(nproc)

python -m pip install \
  --no-cache-dir \
  --force-reinstall \
  --no-build-isolation \
  --no-binary=mmcv \
  "mmcv==2.1.0" "mmsegmentation" -v
