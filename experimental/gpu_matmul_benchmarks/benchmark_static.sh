#!/bin/bash

set -eu

ninja

srcdir="$(dirname "$0")"


src="mfma_f16_static.mlir"
outdir="/tmp/${src}"
rm -rf "${outdir}"
mkdir -p "${outdir}"
echo
echo "Benchmarking $src:"
for s in 128 256 512 1024 2048 4096 8192; do
  sizedsrc="${outdir}/${src}.size${s}.mlir"
  vmfb="${sizedsrc}.vmfb"
  m=$((s / 128))
  n=$((s / 128))
  k=$((s / 32))
  sed "s/@M/${m}/g;s/@N/${n}/g;s/@K/${k}/g" < "${srcdir}/${src}" > "${sizedsrc}"
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
    --input=${m}x${k}x8x4x16x2x4xf16 \
    --input=${n}x${k}x4x2x4x16x2x4xf16 \
    2>/dev/null | grep -o "\bitems_per_second.*" | grep -o '[0-9.k]*' | sed 's/k/e+3/')"
  Tops_per_s="$(echo "print('%.2f' % (${items_per_s} * ${s} * ${s} * ${s} * 2 * 1e-12))" | python)"
  echo "$s,$Tops_per_s"
done

src="matmul_f16_static.mlir"
outdir="/tmp/${src}"
rm -rf "${outdir}"
mkdir -p "${outdir}"
echo
echo "Benchmarking $src (DT):"
for s in 128 256 512 1024 2048 4096 8192; do
  sizedsrc="${outdir}/${src}.size${s}.mlir"
  vmfb="${outdir}/${src}.vmfb"
  m=$s
  n=$s
  k=$s
  sed "s/@M/${m}/g;s/@N/${n}/g;s/@K/${k}/g" < "${srcdir}/${src}" > "${sizedsrc}"
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
    --input=${m}x${k}xf16 \
    --input=${n}x${k}xf16 \
    2>/dev/null | grep -o "\bitems_per_second.*" | grep -o '[0-9.k]*' | sed 's/k/e+3/')"
  Tops_per_s="$(echo "print('%.2f' % (${items_per_s} * ${s} * ${s} * ${s} * 2 * 1e-12))" | python)"
  echo "$s,$Tops_per_s"
done

src="matmul_f16_static.mlir"
outdir="/tmp/${src}"
rm -rf "${outdir}"
mkdir -p "${outdir}"
echo
echo "Benchmarking $src (NON-DT):"
for s in 128 256 512 1024 2048 4096 8192; do
  sizedsrc="${outdir}/${src}.size${s}.mlir"
  vmfb="${outdir}/${src}.vmfb"
  m=$s
  n=$s
  k=$s
  sed "s/@M/${m}/g;s/@N/${n}/g;s/@K/${k}/g" < "${srcdir}/${src}" > "${sizedsrc}"
  tools/iree-compile \
    --iree-hal-target-backends=rocm \
    --iree-hip-target=gfx942 \
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
    --input=${m}x${k}xf16 \
    --input=${n}x${k}xf16 \
    2>/dev/null | grep -o "\bitems_per_second.*" | grep -o '[0-9.k]*' | sed 's/k/e+3/')"
  Tops_per_s="$(echo "print('%.2f' % (${items_per_s} * ${s} * ${s} * ${s} * 2 * 1e-12))" | python)"
  echo "$s,$Tops_per_s"
done




src="mfma_i8_static.mlir"
outdir="/tmp/${src}"
rm -rf "${outdir}"
mkdir -p "${outdir}"
echo
echo "Benchmarking $src:"
for s in 128 256 512 1024 2048 4096 8192; do
  sizedsrc="${outdir}/${src}.size${s}.mlir"
  vmfb="${outdir}/${src}.vmfb"
  m=$((s / 128))
  n=$((s / 128))
  k=$((s / 64))
  sed "s/@M/${m}/g;s/@N/${n}/g;s/@K/${k}/g" < "${srcdir}/${src}" > "${sizedsrc}"
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
    --input=${m}x${k}x8x4x16x2x8xi8 \
    --input=${n}x${k}x4x2x4x16x2x8xi8 \
    2>/dev/null | grep -o "\bitems_per_second.*" | grep -o '[0-9.k]*' | sed 's/k/e+3/')"
  Tops_per_s="$(echo "print('%.2f' % (${items_per_s} * ${s} * ${s} * ${s} * 2 * 1e-12))" | python)"
  echo "$s,$Tops_per_s"
done

src="matmul_i8_static.mlir"
outdir="/tmp/${src}"
rm -rf "${outdir}"
mkdir -p "${outdir}"
echo
echo "Benchmarking $src (DT):"
for s in 128 256 512 1024 2048 4096 8192; do
  sizedsrc="${outdir}/${src}.size${s}.mlir"
  vmfb="${outdir}/${src}.vmfb"
  m=$s
  n=$s
  k=$s
  sed "s/@M/${m}/g;s/@N/${n}/g;s/@K/${k}/g" < "${srcdir}/${src}" > "${sizedsrc}"
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
    --input=${m}x${k}xi8 \
    --input=${n}x${k}xi8 \
    2>/dev/null | grep -o "\bitems_per_second.*" | grep -o '[0-9.k]*' | sed 's/k/e+3/')"
  Tops_per_s="$(echo "print('%.2f' % (${items_per_s} * ${s} * ${s} * ${s} * 2 * 1e-12))" | python)"
  echo "$s,$Tops_per_s"
done

src="matmul_i8_static.mlir"
outdir="/tmp/${src}"
rm -rf "${outdir}"
mkdir -p "${outdir}"
echo
echo "Benchmarking $src (NON-DT):"
for s in 128 256 512 1024 2048 4096 8192; do
  sizedsrc="${outdir}/${src}.size${s}.mlir"
  vmfb="${outdir}/${src}.vmfb"
  m=$s
  n=$s
  k=$s
  sed "s/@M/${m}/g;s/@N/${n}/g;s/@K/${k}/g" < "${srcdir}/${src}" > "${sizedsrc}"
  tools/iree-compile \
    --iree-hal-target-backends=rocm \
    --iree-hip-target=gfx942 \
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
    --input=${m}x${k}xi8 \
    --input=${n}x${k}xi8 \
    2>/dev/null | grep -o "\bitems_per_second.*" | grep -o '[0-9.k]*' | sed 's/k/e+3/')"
  Tops_per_s="$(echo "print('%.2f' % (${items_per_s} * ${s} * ${s} * ${s} * 2 * 1e-12))" | python)"
  echo "$s,$Tops_per_s"
done
