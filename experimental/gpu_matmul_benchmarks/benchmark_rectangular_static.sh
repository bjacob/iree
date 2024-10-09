#!/bin/bash

set -eu

ninja

srcdir="$(dirname "$0")"
src="mfma_i8_static.mlir"
outdir="/tmp/${src}"
rm -rf "${outdir}"
mkdir -p "${outdir}"

function benchmark() {
  m=$1
  n=$2
  k=$3

  sizedsrc="${outdir}/${src}.size${m}x${n}x${k}.mlir"
  vmfb="${sizedsrc}.vmfb"
  mo=$((m / 128))
  no=$((n / 128))
  ko=$((k / 64))
  sed "s/@M/${mo}/g;s/@N/${no}/g;s/@K/${ko}/g" < "${srcdir}/${src}" > "${sizedsrc}"
  tools/iree-compile \
    --iree-hal-target-backends=rocm \
    --iree-hip-target=gfx942 \
    --iree-opt-data-tiling \
    --iree-global-opt-experimental-rocm-data-tiling \
    --iree-global-opt-enable-early-materialization \
    "${sizedsrc}" \
    -o "${vmfb}" \
    --iree-hal-dump-executable-intermediates-to="${outdir}" \
    --iree-hal-dump-executable-sources-to="${outdir}"
  items_per_s="$(tools/iree-benchmark-module \
    "--module=${vmfb}" \
    --benchmark_min_warmup_time=0.1 \
    --device=hip \
    --device_allocator=caching \
    --function=foo \
    --input=${mo}x${ko}x8x4x16x2x8xi8 \
    --input=${no}x${ko}x4x2x4x16x2x8xi8 \
    2>/dev/null | grep -o "\bitems_per_second.*" | grep -o '[0-9.k]*' | sed 's/k/e+3/')"
  Tops_per_s="$(echo "print('%.2f' % (${items_per_s} * ${m} * ${n} * ${k} * 2 * 1e-12))" | python)"
  echo "$m,$n,$k,$Tops_per_s"
}

src="mfma_i8_static.mlir"
outdir="/tmp/${src}"
rm -rf "${outdir}"
mkdir -p "${outdir}"
echo
for s in 128 256 512 1024 2048 4096 8192; do
    benchmark $s $s 128
done
for s in 128 256 512 1024 2048 4096 8192; do
    benchmark $s $s 1024
done
for s in 128 256 512 1024 2048 4096 8192; do
    benchmark $s $s 8192
done
for s in 128 256 512 1024 2048 4096 8192; do
    benchmark 128 128 $s
done
for s in 128 256 512 1024 2048 4096 8192; do
    benchmark 1024 1024 $s
done
for s in 128 256 512 1024 2048 4096 8192; do
    benchmark 8129 8129 $s
done
