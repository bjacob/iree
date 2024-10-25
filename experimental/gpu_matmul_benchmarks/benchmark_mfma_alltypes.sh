#!/bin/bash

set -eu

ninja

srcdir="$(dirname "$0")"

for src in mfma_i8.mlir mfma_f16.mlir mfma_bf16.mlir mfma_f8E5M2FNUZ.mlir mfma_f8E4M3FNUZ.mlir; do
  outdir="/tmp/${src}"
  rm -rf "${outdir}"
  mkdir -p "${outdir}"
  tools/iree-compile \
    --iree-hal-target-backends=rocm \
    --iree-hip-target=gfx942 \
    --iree-opt-data-tiling \
    --iree-global-opt-experimental-rocm-data-tiling \
    --iree-global-opt-enable-early-materialization \
    "${srcdir}/${src}" \
    -o "${outdir}/${src}.vmfb" \
    --iree-hal-dump-executable-intermediates-to="${outdir}" \
    --iree-hal-dump-executable-sources-to="${outdir}"
done

src=mfma_f16.mlir
echo
echo "Benchmarking $src:"
for s in 4096; do
  m=$((s / 128))
  n=$((s / 128))
  k=$((s / 32))
  items_per_s="$(tools/iree-benchmark-module \
    "--module=/tmp/${src}/${src}.vmfb" \
    --benchmark_min_warmup_time=0.5 \
    --device=hip \
    --device_allocator=caching \
    --function=foo \
    --input=${m}x${k}x8x4x16x2x4xf16 \
    --input=${n}x${k}x4x2x4x16x2x4xf16 \
    2>/dev/null | grep -o "\bitems_per_second.*" | grep -o '[0-9.k]*' | sed 's/k/e+3/')"
  Tops_per_s="$(echo "print('%.2f' % (${items_per_s} * ${s} * ${s} * ${s} * 2 * 1e-12))" | python)"
  echo "$s,$Tops_per_s"
done

src=mfma_bf16.mlir
echo
echo "Benchmarking $src:"
for s in 4096; do
  m=$((s / 128))
  n=$((s / 128))
  k=$((s / 32))
  items_per_s="$(tools/iree-benchmark-module \
    "--module=/tmp/${src}/${src}.vmfb" \
    --benchmark_min_warmup_time=0.5 \
    --device=hip \
    --device_allocator=caching \
    --function=foo \
    --input=${m}x${k}x8x4x16x2x4xbf16 \
    --input=${n}x${k}x4x2x4x16x2x4xbf16 \
    2>/dev/null | grep -o "\bitems_per_second.*" | grep -o '[0-9.k]*' | sed 's/k/e+3/')"
  Tops_per_s="$(echo "print('%.2f' % (${items_per_s} * ${s} * ${s} * ${s} * 2 * 1e-12))" | python)"
  echo "$s,$Tops_per_s"
done

src=mfma_i8.mlir
echo
echo "Benchmarking $src:"
for s in 4096; do
  m=$((s / 128))
  n=$((s / 128))
  k=$((s / 64))
  items_per_s="$(tools/iree-benchmark-module \
    "--module=/tmp/${src}/${src}.vmfb" \
    --device=hip \
    --device_allocator=caching \
    --function=foo \
    --input=${m}x${k}x8x4x16x2x8xi8 \
    --input=${n}x${k}x4x2x4x16x2x8xi8 \
    2>/dev/null | grep -o "\bitems_per_second.*" | grep -o '[0-9.k]*' | sed 's/k/e+3/')"
  Tops_per_s="$(echo "print('%.2f' % (${items_per_s} * ${s} * ${s} * ${s} * 2 * 1e-12))" | python)"
  echo "$s,$Tops_per_s"
done

src=mfma_f8E5M2FNUZ.mlir
echo
echo "Benchmarking $src:"
for s in 4096; do
  m=$((s / 128))
  n=$((s / 128))
  k=$((s / 64))
  items_per_s="$(tools/iree-benchmark-module \
    "--module=/tmp/${src}/${src}.vmfb" \
    --device=hip \
    --device_allocator=caching \
    --function=foo \
    --input=${m}x${k}x8x4x16x2x8xf8E5M2FNUZ \
    --input=${n}x${k}x4x2x4x16x2x8xf8E5M2FNUZ \
    2>/dev/null | grep -o "\bitems_per_second.*" | grep -o '[0-9.k]*' | sed 's/k/e+3/')"
  Tops_per_s="$(echo "print('%.2f' % (${items_per_s} * ${s} * ${s} * ${s} * 2 * 1e-12))" | python)"
  echo "$s,$Tops_per_s"
done

src=mfma_f8E4M3FNUZ.mlir
echo
echo "Benchmarking $src:"
for s in 4096; do
  m=$((s / 128))
  n=$((s / 128))
  k=$((s / 64))
  items_per_s="$(tools/iree-benchmark-module \
    "--module=/tmp/${src}/${src}.vmfb" \
    --device=hip \
    --device_allocator=caching \
    --function=foo \
    --input=${m}x${k}x8x4x16x2x8xf8E4M3FNUZ \
    --input=${n}x${k}x4x2x4x16x2x8xf8E4M3FNUZ \
    2>/dev/null | grep -o "\bitems_per_second.*" | grep -o '[0-9.k]*' | sed 's/k/e+3/')"
  Tops_per_s="$(echo "print('%.2f' % (${items_per_s} * ${s} * ${s} * ${s} * 2 * 1e-12))" | python)"
  echo "$s,$Tops_per_s"
done
