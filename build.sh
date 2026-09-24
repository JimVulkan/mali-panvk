#!/bin/sh
# Build the driver for Android (arm64, API 31) and package it into dist/.
#
# usage: ./build.sh [path-to-android-ndk]
#   The NDK can also come from ANDROID_NDK_HOME or ANDROID_NDK_ROOT. Linux only (WSL works): the
#   first stage builds host tools that need LLVM.
#   BUILD_DIR overrides the Android build directory (default: build-android).
#
# Two stages:
#   1. build-host: mesa_clc, vtn_bindgen2 and panfrost_compile, native, from this tree. They
#      compile the driver's OpenCL helper kernels (libpan, poly) at build time, so they have to be
#      built from the same compiler sources as the driver.
#   2. build-android: libvulkan_panfrost.so, cross-compiled with the NDK, using those tools.
#
# Requires: meson, ninja, python3 (mako, packaging, pyyaml), pkg-config, glslangValidator, and for
# stage 1 LLVM with clang, libclc and the SPIR-V LLVM translator of one major version (Ubuntu:
# llvm-dev libclang-dev libclc-XX-dev libllvmspirvlib-XX-dev clang).
set -eu

cd "$(dirname "$0")"

NDK="${1:-${ANDROID_NDK_HOME:-${ANDROID_NDK_ROOT:-}}}"
if [ -z "$NDK" ] || [ ! -d "$NDK" ]; then
   echo "error: pass the Android NDK path or set ANDROID_NDK_HOME" >&2
   exit 1
fi

case "$(uname -s)" in
   Linux*) HOST=linux-x86_64 ;;
   *) echo "error: build on Linux (or WSL); stage 1 needs LLVM" >&2; exit 1 ;;
esac

BIN="$NDK/toolchains/llvm/prebuilt/$HOST/bin"
if [ ! -x "$BIN/aarch64-linux-android31-clang" ]; then
   echo "error: no API 31 arm64 compiler in $BIN" >&2
   exit 1
fi

for tool in meson ninja python3 pkg-config glslangValidator; do
   if ! command -v "$tool" >/dev/null 2>&1; then
      echo "error: $tool not found in PATH" >&2
      exit 1
   fi
done

# --- stage 1: host tools ----------------------------------------------------------------------
HOSTB=build-host
if [ ! -f "$HOSTB/build.ninja" ]; then
   meson setup "$HOSTB" \
      -Dbuildtype=debugoptimized \
      -Dmesa-clc=enabled -Dprecomp-compiler=enabled \
      -Dvulkan-drivers=panfrost -Dgallium-drivers= -Dplatforms= -Dllvm=enabled \
      -Dgles1=disabled -Dgles2=disabled -Degl=disabled -Dglx=disabled -Dgbm=disabled \
      -Dtools= -Dvideo-codecs= \
      -Dallow-fallback-for=libdrm --force-fallback-for=libdrm,expat,zlib \
      -Dlibdrm:default_library=static
fi
ninja -C "$HOSTB" src/compiler/clc/mesa_clc src/compiler/spirv/vtn_bindgen2 \
   src/panfrost/clc/panfrost_compile
mkdir -p "$HOSTB/bin"
cp "$HOSTB/src/compiler/clc/mesa_clc" "$HOSTB/src/compiler/spirv/vtn_bindgen2" \
   "$HOSTB/src/panfrost/clc/panfrost_compile" "$HOSTB/bin/"
PATH="$PWD/$HOSTB/bin:$PATH"
export PATH

# --- stage 2: the driver ----------------------------------------------------------------------
BUILD="${BUILD_DIR:-build-android}"
mkdir -p "$BUILD" "$BUILD/pkgconfig-empty"

cat > "$BUILD/cross.ini" <<EOF
[binaries]
c = '$BIN/aarch64-linux-android31-clang'
cpp = '$BIN/aarch64-linux-android31-clang++'
ar = '$BIN/llvm-ar'
strip = '$BIN/llvm-strip'
pkg-config = 'pkg-config'

[properties]
# Keep the host's .pc files out of the cross build.
pkg_config_libdir = ['$PWD/$BUILD/pkgconfig-empty']
cpp_link_args = ['-static-libstdc++']

[host_machine]
system = 'android'
cpu_family = 'aarch64'
cpu = 'aarch64'
endian = 'little'
EOF

if [ ! -f "$BUILD/build.ninja" ]; then
   meson setup "$BUILD" --cross-file "$BUILD/cross.ini" \
      -Dbuildtype=debugoptimized -Db_ndebug=true \
      -Dplatforms=android -Dplatform-sdk-version=31 -Dandroid-stub=true -Dandroid-strict=false \
      -Dandroid-libbacktrace=disabled \
      -Dvulkan-drivers=panfrost -Dgallium-drivers= -Dtools= -Dvideo-codecs= \
      -Dmesa-clc=system -Dprecomp-compiler=system \
      -Dllvm=disabled -Dzstd=disabled -Dlmsensors=disabled -Dperfetto=false \
      -Dglx=disabled -Dgbm=disabled -Degl=disabled -Dgles1=disabled -Dgles2=disabled \
      -Dallow-fallback-for=libdrm,perfetto --force-fallback-for=expat,libdrm,zlib \
      -Dlibdrm:default_library=static -Dexpat:default_library=static -Dzlib:default_library=static
fi

ninja -C "$BUILD" src/panfrost/vulkan/libvulkan_panfrost.so

mkdir -p dist
"$BIN/llvm-strip" -o "$BUILD/libvulkan_panfrost.so" "$BUILD/src/panfrost/vulkan/libvulkan_panfrost.so"
python3 android/package.py "$BUILD/libvulkan_panfrost.so" dist
