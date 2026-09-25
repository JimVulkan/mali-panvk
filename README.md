# PanVK for Mali Bifrost on kbase

A port of Mesa's PanVK Vulkan driver to Arm Mali Bifrost GPUs on stock Android, based on Mesa
26.3.0-devel. It runs on the kernel driver the phone already has (Arm's kbase, `/dev/mali0`), so
it needs no root and no custom kernel.

Tested on the Mali-G76 (Galaxy S10e, Exynos 9820), the Mali-G72 (Galaxy Tab S6 Lite, Exynos 9611)
and the Mali-G52 (Galaxy A31, MediaTek Helio P65).

## What works

- Vulkan 1.3 as PanVK exposes it on Bifrost, through the kbase job interface.
- Geometry shaders, tessellation and transform feedback, which Bifrost does not have in hardware,
  emulated with compute jobs. Indirect, indexed, instanced and multi-draw calls work with all of
  them.
- `VK_EXT_robustness2` (out-of-range and null descriptor accesses read zero) and geometry
  shader vertex streams, which DXVK requires for Direct3D 9, 10 and 11. BCn textures still have
  to come from the app: Winlator-based emulators decode them in their Vulkan wrapper.
- Presentation through Android's gralloc (Arm's handle layout, including MediaTek's).

Not done yet: BCn texture formats, and what vkd3d-proton needs for Direct3D 12.

On phones with 4 GB of RAM or less, give Minecraft about 600 MB of Java heap: GPU memory comes out
of the same RAM, and with a 1 GB heap the system kills the game while it loads a world.

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

No Android platform libraries are needed: the build uses Mesa's Android stubs.

## Building

```sh
./build.sh /path/to/android-ndk
```

The NDK path can also come from `ANDROID_NDK_HOME` or `ANDROID_NDK_ROOT`. The script first builds
three host tools into `build-host/` (`mesa_clc`, `vtn_bindgen2`, `panfrost_compile`: the driver's
OpenCL helper kernels are compiled with them, so they come from this tree). It then builds the
driver into `build-android/` (set `BUILD_DIR` to change it), strips it, and writes a package to
`dist/`: a zip with `meta.json`, `libvulkan_panfrost.so` and `NOTICE.txt`, for emulators and
launchers that load custom Vulkan drivers from a zip.

## Runtime switches

| Environment | Effect |
|---|---|
| `PANVK_DEBUG=startup` | logs why the device could not be created |
| `PANVK_FORCE_IDVS=1` | turns index-driven vertex shading back on (off by default on Bifrost) |

## Disclaimer

This project was developed with heavy use of AI tools. Every change is built and tested on the
devices above before it is published.

## License

The changes are MIT licensed (`LICENSE`). Files from Mesa keep the license in their headers,
mostly MIT; `THIRD_PARTY_NOTICES.md` lists the files under other licenses and `LICENSES/` has the
full texts. The notices for everything compiled into the driver ship in the package as
`NOTICE.txt`.

This project is not affiliated with or endorsed by Arm, Samsung, MediaTek or the Mesa project.
Arm and Mali are trademarks of Arm Limited. Samsung, Galaxy and Exynos are trademarks of Samsung
Electronics. MediaTek and Helio are trademarks of MediaTek Inc.
