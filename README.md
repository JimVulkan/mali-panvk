# PanVK for Mali on kbase

A port of Mesa's PanVK Vulkan driver to Arm Mali GPUs on stock Android, based on Mesa
27.0.0-devel. It runs on the kernel driver the phone already has (Arm's kbase, `/dev/mali0`), so it
needs no root and no custom kernel.

## Supported GPUs

| Architecture | GPUs | Vulkan | Status |
|---|---|---|---|
| Bifrost (job manager) | Mali-G31, G51, G52, G71, G72, G76 | 1.4 (1.3 on GPUs with fewer than 8 colour attachments, such as the G71) | tested on G76, G72, G52 |
| Valhall, first generation (job manager) | Mali-G57, G68, G77, G78 | 1.4 | tested on G57, G68, G77, G78 |
| Valhall and 5th generation with the command stream frontend | Mali-G610, G710, G615, G715, G720, Immortalis-G925, Mali-G1-Ultra | 1.4 | tested on G720, G925 and G1-Ultra, and on a G615 by a tester; G610, G710 and G715 untested |

Devices it was tested on: Galaxy S10e (G76, Exynos 9820), Galaxy Tab S6 Lite (G72, Exynos 9611),
Galaxy A31 (G52, MediaTek Helio P65), Galaxy A15 and Tab A11 (G57, MediaTek Helio G99), Galaxy A35,
Tab S9 FE and Tab S9 FE+ (G68, Exynos 1380), Galaxy S20 and S20 Ultra (G77, Exynos 990), Galaxy
S21 (G78, Exynos 2100), Galaxy Tab S10+ and Tab S10 Ultra (Immortalis-G720, MediaTek Dimensity
9300+), Galaxy Tab S11 (Immortalis-G925, MediaTek Dimensity 9400+), Galaxy Tab S12+ (Mali-G1-Ultra,
MediaTek), and a tester's G615 phone.

## What works

- Vulkan 1.4 through the kbase job or command stream interface (1.3 on the smallest Bifrost GPUs).
- Geometry shaders, tessellation and transform feedback, which these GPUs do not have in hardware,
  emulated with compute jobs. Indirect, indexed, instanced and multi-draw calls work with them,
  and so do transform feedback queries.
- `VK_EXT_robustness2` with null descriptors (which DXVK and zink require) and geometry shader
  vertex streams: Direct3D 9, 10 and 11 through DXVK, and OpenGL through zink.
- Direct3D 12: on Valhall, everything vkd3d-proton requires, plus `VK_EXT_descriptor_heap`; on
  Bifrost, Wine's own d3d12 (upstream vkd3d) at feature level 11_0.
- BCn textures: emulated where the GPU does not decode them (BC4-BC7 on the G52, which decodes
  BC1-BC3; all of them on the G76, which decodes none), native where it does.
- Timestamp queries on every GPU, and pipeline statistics queries (counted on the CPU, for direct
  draws and dispatches).
- 16-bit arithmetic for `mediump` / `RelaxedPrecision` shader code.
- A disk shader cache, kept next to the driver (or in `$TMPDIR`), so pipelines are compiled once.
- Presentation through Android's gralloc and hardware buffers shared with other apps, including
  MediaTek's AFBC-compressed ones (see [Gralloc](#gralloc)).

Not done yet: sparse resources (kbase cannot remap GPU pages, so Direct3D 12 stops at feature level
11_1), and mesh shaders (no Bifrost or Valhall GPU has them in hardware).

On phones with 4 GB of RAM or less, give Minecraft about 600 MB of Java heap: GPU memory comes out
of the same RAM, and with a 1 GB heap the system kills the game while it loads a world.

## Gralloc

To present, and to import buffers from other apps, the driver has to know how each buffer is laid
out. It asks the device's gralloc HAL through libui's `GraphicBufferMapper` (any mapper from
IMapper 2.0 to IMapper 5), and otherwise reads Arm's and MediaTek's buffer handles directly.

The gralloc code is Mesa's, but it calls Android platform libraries the NDK does not include. The
build therefore needs their headers and something to link against:

- The headers are in `android/gralloc/include`: the 66 AOSP headers the build reads, under the
  Apache License 2.0 (one of them cut down; `android/gralloc/NOTICE.md` lists where each comes
  from).
- Nothing from a device is needed. `build.sh` generates empty stand-in libraries with the same
  names and symbols (`android/gralloc/stubs.txt`, `mkstubs.py`), and the driver links against
  those. On the phone, Android loads its own libraries.

The driver links against `liblog.so`, `libnativewindow.so` and the platform `libc++.so`, which
every Android 12+ phone has. `libui.so` is not linked: the driver opens it at run time, so it
still loads in launchers that keep it in the vendor driver's linker namespace (AdrenoTools,
Bannerlator, Winlator), where libui is not visible. There it uses the handle readers.
`GRALLOC=0 ./build.sh` builds without the gralloc backend at all.

## Requirements

Tools:

| Tool | Version | Notes |
|---|---|---|
| Linux or WSL | | the host tools need LLVM |
| Android NDK | r29 tested | the API 31 aarch64 compiler and `llvm-strip` |
| Meson | 1.4.0 or newer | |
| Ninja | any recent | |
| Python | 3.10 or newer | with `mako`, `packaging` and `pyyaml` |
| glslangValidator | 12.2 or newer | from the Vulkan SDK or your distribution |
| LLVM, clang, libclc, SPIR-V LLVM translator | one major version, 21 tested | for the host tools |

On Ubuntu:

```sh
sudo apt install meson ninja-build glslang-tools pkg-config python3-mako python3-packaging \
    python3-yaml llvm-21-dev libclang-21-dev clang-21 libclc-21-dev libllvmspirvlib-21-dev
```

Dependencies are fetched by Meson on the first configure, so that step needs network access:

| Library | Version | Use |
|---|---|---|
| libdrm | 2.4.133 | linked statically |
| zlib | 1.3.1 | linked statically |
| Expat | 2.5.0 | required by the configure step, not linked into the driver |

## Building

```sh
./build.sh /path/to/android-ndk
```

The NDK path can also come from `ANDROID_NDK_HOME` or `ANDROID_NDK_ROOT`. The script first builds
three host tools into `build-host/` (`mesa_clc`, `vtn_bindgen2`, `panfrost_compile`: the driver's
OpenCL helper kernels are compiled with them, so they come from this tree). It then generates the
gralloc link stubs, builds the driver into `build-android/` (set `BUILD_DIR` to change it), strips
it, and writes a package to `dist/`: a zip with `meta.json`, `libvulkan_panfrost.so` and
`NOTICE.txt`, for emulators and launchers that load custom Vulkan drivers from a zip.

## Runtime switches

| Environment | Effect |
|---|---|
| `PANVK_DEBUG=startup` | logs why the device could not be created |
| `PANVK_FORCE_IDVS=1` | turns index-driven vertex shading back on (off by default on Bifrost) |
| `PANVK_NO_MEDIUMP=1` | compiles `mediump` shader code at 32 bits |
| `MESA_SHADER_CACHE_DISABLE=true` | turns the disk shader cache off |
| `MESA_LOG=file` | writes the driver's log, including the details of a GPU fault, to a file |

## Disclaimer

This project was developed with heavy use of AI tools. Every change is built and tested on the
devices above before it is published.

## License

The changes are MIT licensed (`LICENSE`). Files from Mesa keep the license in their headers,
mostly MIT; `THIRD_PARTY_NOTICES.md` lists the files under other licenses and `LICENSES/` has the
full texts. `android/gralloc/include` is Apache-2.0 (The Android Open Source Project). The notices
for everything compiled into the driver ship in the package as `NOTICE.txt`.

This project is not affiliated with or endorsed by Arm, Samsung, MediaTek, Google or the Mesa
project. Arm, Mali and Immortalis are trademarks of Arm Limited. Samsung, Galaxy and Exynos are
trademarks of Samsung Electronics. MediaTek, Helio and Dimensity are trademarks of MediaTek Inc.
Android is a trademark of Google LLC.
