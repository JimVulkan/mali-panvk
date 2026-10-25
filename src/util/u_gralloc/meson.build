# Mesa 3-D graphics library
#
# Copyright (C) 2022 Roman Stratiienko (r.stratiienko@gmail.com)
# SPDX-License-Identifier: MIT

c_args_for_u_gralloc = []
cpp_args_for_u_gralloc = []
options_for_u_gralloc = []
deps_for_u_gralloc = [dep_android]

files_u_gralloc = files(
  'u_gralloc.c',
  'u_gralloc_internal.c',
  'u_gralloc_fallback.c',
  'u_gralloc_cros_api.c',
  'u_gralloc_libdrm.c',
  'u_gralloc_qcom.c',
)

if dep_android_ui.found()
  files_u_gralloc += files('u_gralloc_imapper5_api.cpp')
  c_args_for_u_gralloc += '-DUSE_IMAPPER4_METADATA_API'
  cpp_args_for_u_gralloc += '-DUSE_IMAPPER4_METADATA_API'
  options_for_u_gralloc += 'cpp_std=c++17'
  deps_for_u_gralloc += dep_android_ui
elif dep_android_mapper4.found()
  files_u_gralloc += files('u_gralloc_imapper4_api.cpp')
  c_args_for_u_gralloc += '-DUSE_IMAPPER4_METADATA_API'
  cpp_args_for_u_gralloc += '-DUSE_IMAPPER4_METADATA_API'
  # XCLIPSE: the platform libs this file calls (libgralloctypes, mapper@4.0, libhidlbase) are built
  # with the platform libc++, whose inline namespace is std::__1; the NDK's is std::__ndk1, so no
  # std::vector/std::string-carrying symbol would link. This one file is compiled against a copy
  # of the NDK's __config_site with __1, ahead of the sysroot (a plain -D loses: __config_site
  # defines the namespace unconditionally). Safe because u_gralloc's boundary to the rest of Mesa
  # is C structs only; do not use it anywhere else.
  if get_option('xclipse-gralloc-deps') != ''
    cpp_args_for_u_gralloc += ['-isystem', get_option('xclipse-gralloc-deps') / 'abi_override/c++/v1']
  endif
  options_for_u_gralloc += 'cpp_std=c++17'
  deps_for_u_gralloc += dep_android_mapper4
endif

if with_freedreno_vk or with_gallium_freedreno
  c_args_for_u_gralloc += '-DHAS_FREEDRENO'
endif

# XCLIPSE: read Samsung gralloc handles directly (RADV WSI, radeonsi EGL window surfaces).
if with_amd_vk or with_gallium_radeonsi
  c_args_for_u_gralloc += '-DHAS_SAMSUNG_GRALLOC'
endif

_libmesa_u_gralloc = static_library(
  '_mesa_u_gralloc',
  [files_u_gralloc],
  include_directories : [inc_include, inc_util],
  c_args : c_args_for_u_gralloc,
  cpp_args : cpp_args_for_u_gralloc,
  override_options : options_for_u_gralloc,
  gnu_symbol_visibility : 'hidden',
  build_by_default : false,
  dependencies: deps_for_u_gralloc,
)

idep_u_gralloc = declare_dependency(
  link_with: _libmesa_u_gralloc,
  dependencies: deps_for_u_gralloc,
)
