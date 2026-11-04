/*
 * Mesa 3-D graphics library
 *
 * Copyright (C) 2026 NXP
 * Copyright (C) 2021 GlobalLogic Ukraine
 * Copyright (C) 2021-2022 Roman Stratiienko (r.stratiienko@gmail.com)
 * SPDX-License-Identifier: MIT
 */
#include <aidl/android/hardware/graphics/common/BufferUsage.h>
#include <aidl/android/hardware/graphics/common/ChromaSiting.h>
#include <aidl/android/hardware/graphics/common/Dataspace.h>
#include <aidl/android/hardware/graphics/common/ExtendableType.h>
#include <aidl/android/hardware/graphics/common/PlaneLayoutComponent.h>
#include <aidl/android/hardware/graphics/common/PlaneLayoutComponentType.h>
#include <ui/GraphicBufferMapper.h>

#include <system/window.h>
#include <dlfcn.h>
#include <new>
#include <string.h>
#include <unistd.h>

#include "util/log.h"
#include "u_gralloc_internal.h"

using aidl::android::hardware::graphics::common::BufferUsage;
using aidl::android::hardware::graphics::common::ChromaSiting;
using aidl::android::hardware::graphics::common::ExtendableType;
using aidl::android::hardware::graphics::common::PlaneLayout;
using aidl::android::hardware::graphics::common::PlaneLayoutComponent;
using aidl::android::hardware::graphics::common::PlaneLayoutComponentType;
using android::hardware::graphics::common::V1_2::Dataspace;
using android::GraphicBufferMapper;
using android::OK;
using android::status_t;

/* XCLIPSE: libui is opened at run time, not linked. AdrenoTools (Winlator, Bannerlator) loads
 * the driver in the vendor driver's linker namespace, where libui.so is not visible, and a
 * NEEDED entry on it made the whole driver fail to load there. Without it this backend is
 * skipped and the handle parsers (u_gralloc_fallback.c) describe the buffers instead.
 *
 * The methods are plain (non-virtual) members, so each is called through its exported symbol
 * with the object as the first argument, which is how AArch64 passes "this". get() is an inline
 * Singleton in the header; libui exports the singleton's lock and instance, so it is redone
 * here the same way. */
struct libui_api {
   GraphicBufferMapper *mapper;
   status_t (*getPixelFormatFourCC)(GraphicBufferMapper *, buffer_handle_t, uint32_t *);
   status_t (*getPixelFormatModifier)(GraphicBufferMapper *, buffer_handle_t, uint64_t *);
   status_t (*getAllocationSize)(GraphicBufferMapper *, buffer_handle_t, uint64_t *);
   status_t (*getLayerCount)(GraphicBufferMapper *, buffer_handle_t, uint64_t *);
   status_t (*getPlaneLayouts)(GraphicBufferMapper *, buffer_handle_t, std::vector<PlaneLayout> *);
   status_t (*getChromaSiting)(GraphicBufferMapper *, buffer_handle_t, ChromaSiting *);
   status_t (*getDataspace)(GraphicBufferMapper *, buffer_handle_t, Dataspace *);
};

static libui_api ui;

static bool
libui_open(void)
{
   void *lib = dlopen("libui.so", RTLD_NOW | RTLD_LOCAL);
   if (!lib) {
      mesa_logi("libui.so is not visible here (%s); using the gralloc handle parsers", dlerror());
      return false;
   }

#define UI_SYM(field, name)                                                                    \
   *(void **)&ui.field = dlsym(lib, name);                                                     \
   if (!ui.field)                                                                              \
      goto missing;

   UI_SYM(getPixelFormatFourCC, "_ZN7android19GraphicBufferMapper20getPixelFormatFourCCEPK13native_handlePj");
   UI_SYM(getPixelFormatModifier, "_ZN7android19GraphicBufferMapper22getPixelFormatModifierEPK13native_handlePm");
   UI_SYM(getAllocationSize, "_ZN7android19GraphicBufferMapper17getAllocationSizeEPK13native_handlePm");
   UI_SYM(getLayerCount, "_ZN7android19GraphicBufferMapper13getLayerCountEPK13native_handlePm");
   UI_SYM(getPlaneLayouts,
          "_ZN7android19GraphicBufferMapper15getPlaneLayoutsEPK13native_handlePNSt3__16vectorIN4aidl7android8"
          "hardware8graphics6common11PlaneLayoutENS4_9allocatorISB_EEEE");
   UI_SYM(getChromaSiting,
          "_ZN7android19GraphicBufferMapper15getChromaSitingEPK13native_handlePN4aidl7android8hardware8graphics6"
          "common12ChromaSitingE");
   UI_SYM(getDataspace,
          "_ZN7android19GraphicBufferMapper12getDataspaceEPK13native_handlePNS_8hardware8graphics6common4V1_29"
          "DataspaceE");
#undef UI_SYM

   {
      auto *lock = (android::Mutex *)dlsym(lib, "_ZN7android9SingletonINS_19GraphicBufferMapperEE5sLockE");
      auto **instance =
         (GraphicBufferMapper **)dlsym(lib, "_ZN7android9SingletonINS_19GraphicBufferMapperEE9sInstanceE");
      void (*ctor)(GraphicBufferMapper *);
      *(void **)&ctor = dlsym(lib, "_ZN7android19GraphicBufferMapperC1Ev");
      if (!lock || !instance || !ctor)
         goto missing;

      lock->lock();
      if (!*instance) {
         /* Room to spare: a newer libui's object may be larger than this header's. */
         void *mem = ::operator new(sizeof(GraphicBufferMapper) < 256 ? 256 : sizeof(GraphicBufferMapper));
         memset(mem, 0, sizeof(GraphicBufferMapper) < 256 ? 256 : sizeof(GraphicBufferMapper));
         ctor((GraphicBufferMapper *)mem);
         *instance = (GraphicBufferMapper *)mem;
      }
      ui.mapper = *instance;
      lock->unlock();
   }
   return true;

missing:
   mesa_logw("libui.so lacks a GraphicBufferMapper symbol; using the gralloc handle parsers");
   memset(&ui, 0, sizeof(ui));
   dlclose(lib);
   return false;
}

static std::optional<std::vector<PlaneLayout>>
GetPlaneLayouts(const native_handle_t *buffer)
{
   std::vector<PlaneLayout> plane_layouts;
   status_t  error = ui.getPlaneLayouts(ui.mapper, buffer, &plane_layouts);

   if (error != OK)
      return std::nullopt;

   return plane_layouts;
}

struct gralloc_mapper {
   struct u_gralloc base;
   /* XCLIPSE: handles the mapper cannot describe go to the handle parsers instead. */
   struct u_gralloc *fallback;
};

extern "C" {

/* Whether fd is something with a size (a dma-buf), leaving its offset at 0. */
static bool
mapper5_fd_has_size(int fd)
{
   const off_t sz = lseek(fd, 0, SEEK_END);
   lseek(fd, 0, SEEK_SET);
   return sz > 0;
}

static int
mapper5_query_buffer_basic_info(struct u_gralloc *gralloc,
                                struct u_gralloc_buffer_handle *hnd,
                                struct u_gralloc_buffer_basic_info *out)
{
   if (!hnd->handle)
      return -EINVAL;

   uint32_t drm_fourcc;
   status_t error = ui.getPixelFormatFourCC(ui.mapper, hnd->handle, &drm_fourcc);
   if (error != OK)
      return -EINVAL;

   uint64_t modifier;
   error = ui.getPixelFormatModifier(ui.mapper, hnd->handle, &modifier);
   if (error != OK)
      return  -EINVAL;

   uint64_t alloc_size;
   error = ui.getAllocationSize(ui.mapper, hnd->handle, &alloc_size);
   if (error != OK)
      return -EINVAL;

   uint64_t layer_count;
   error = ui.getLayerCount(ui.mapper, hnd->handle, &layer_count);
   if (error != OK)
      return -EINVAL;

   out->drm_fourcc = drm_fourcc;
   out->modifier = modifier;
   out->alloc_size = alloc_size;
   out->layer_count = layer_count;

   auto layouts_opt = GetPlaneLayouts(hnd->handle);

   if (!layouts_opt)
      return  -EINVAL;

   std::vector<PlaneLayout> &layouts = *layouts_opt;

   out->num_planes = layouts.size();

   int fd_index = 0;

   /* XCLIPSE: some gralloc4 implementations (Samsung's) leave strideInBytes at 0 on linear
    * buffers while every other field is right. Rebuild it from the plane's width and component
    * sizes, or from its total size when it is the only plane. */
   for (uint32_t i = 0; i < layouts.size(); i++) {
      if (layouts[i].strideInBytes != 0)
         continue;

      int64_t bits = 0;
      for (const auto &comp : layouts[i].components)
         bits += comp.sizeInBits;

      int64_t stride = 0;
      if (bits > 0 && bits % 8 == 0 && layouts[i].widthInSamples > 0)
         stride = layouts[i].widthInSamples * (bits / 8);
      else if (layouts.size() == 1 && layouts[i].heightInSamples > 0)
         stride = layouts[i].totalSizeInBytes / layouts[i].heightInSamples;
      if (stride <= 0)
         return -EINVAL;
      layouts[i].strideInBytes = stride;
   }

   for (uint32_t i = 0; i < layouts.size(); i++) {
      out->strides[i] = layouts[i].strideInBytes;
      out->offsets[i] = layouts[i].offsetInBytes;

      /* offset == 0 means layer is located in different dma-buf */
      if (out->offsets[i] == 0 && i > 0)
         fd_index++;

      if (fd_index >= hnd->handle->numFds)
         return -EINVAL;

      /* XCLIPSE: MediaTek's ARM gralloc puts a -1 fd ahead of the buffer's, or (Dimensity 9300,
       * Android 15) an fd that is not a dma-buf at all: lseek gives it no size. Skip both. */
      while (fd_index < hnd->handle->numFds &&
             (hnd->handle->data[fd_index] < 0 || !mapper5_fd_has_size(hnd->handle->data[fd_index])))
         fd_index++;
      if (fd_index >= hnd->handle->numFds)
         return -EINVAL;

      out->fds[i] = hnd->handle->data[fd_index];
   }

   return 0;
}

static int
mapper5_get_buffer_basic_info(struct u_gralloc *gralloc,
                              struct u_gralloc_buffer_handle *hnd,
                              struct u_gralloc_buffer_basic_info *out)
{
   gralloc_mapper *gr = (gralloc_mapper *)gralloc;

   int ret = mapper5_query_buffer_basic_info(gralloc, hnd, out);
   if (ret == 0 || !gr->fallback || !hnd->handle)
      return ret;

   memset(out, 0, sizeof(*out));
   return gr->fallback->ops.get_buffer_basic_info(gr->fallback, hnd, out);
}

static int
mapper5_get_buffer_color_info(struct u_gralloc *gralloc,
                              struct u_gralloc_buffer_handle *hnd,
                              struct u_gralloc_buffer_color_info *out)
{
   if (!hnd->handle)
      return  -EINVAL;

   /* optional attributes */
   ChromaSiting chroma_siting;
   status_t error = ui.getChromaSiting(ui.mapper, hnd->handle, &chroma_siting);
   if (error != OK)
      return  -EINVAL;

   Dataspace dataspace;
   error = ui.getDataspace(ui.mapper, hnd->handle, &dataspace);
   if (error != OK)
      return  -EINVAL;

   /* default to __DRI_YUV_COLOR_SPACE_ITU_REC601 */
   Dataspace standard =
      (Dataspace)((int)dataspace & (uint32_t)Dataspace::STANDARD_MASK);
   switch (standard) {
      case Dataspace::STANDARD_BT709:
         out->yuv_color_space = __DRI_YUV_COLOR_SPACE_ITU_REC709;
         break;
      case Dataspace::STANDARD_BT2020:
      case Dataspace::STANDARD_BT2020_CONSTANT_LUMINANCE:
         out->yuv_color_space = __DRI_YUV_COLOR_SPACE_ITU_REC2020;
         break;
      case Dataspace::STANDARD_BT601_625:
      case Dataspace::STANDARD_BT601_625_UNADJUSTED:
      case Dataspace::STANDARD_BT601_525:
      case Dataspace::STANDARD_BT601_525_UNADJUSTED:
      default:
         out->yuv_color_space = __DRI_YUV_COLOR_SPACE_ITU_REC601;
         break;
      }

      /* default to __DRI_YUV_NARROW_RANGE */
      Dataspace range =
         (Dataspace)((int)dataspace & (uint32_t)Dataspace::RANGE_MASK);
      switch (range) {
         case Dataspace::RANGE_FULL:
            out->sample_range = __DRI_YUV_FULL_RANGE;
            break;
         case Dataspace::RANGE_LIMITED:
         default:
            out->sample_range = __DRI_YUV_NARROW_RANGE;
            break;
      }

      /* default to __DRI_YUV_CHROMA_SITING_0_5 */
      switch (chroma_siting) {
      case ChromaSiting::COSITED_HORIZONTAL:
         out->horizontal_siting = __DRI_YUV_CHROMA_SITING_0;
         out->vertical_siting = __DRI_YUV_CHROMA_SITING_0_5;
         break;
      case ChromaSiting::COSITED_VERTICAL:
         out->horizontal_siting = __DRI_YUV_CHROMA_SITING_0_5;
         out->vertical_siting = __DRI_YUV_CHROMA_SITING_0;
         break;
      case ChromaSiting::COSITED_BOTH:
         out->horizontal_siting = __DRI_YUV_CHROMA_SITING_0;
         out->vertical_siting = __DRI_YUV_CHROMA_SITING_0;
         break;
      case ChromaSiting::SITED_INTERSTITIAL:
      default:
         out->horizontal_siting = __DRI_YUV_CHROMA_SITING_0_5;
         out->vertical_siting = __DRI_YUV_CHROMA_SITING_0_5;
         break;
      }

   return 0;
}

static int
mapper5_get_front_rendering_usage(struct u_gralloc *gralloc,
                                  uint64_t *out_usage)
{
   assert(out_usage);
#if ANDROID_API_LEVEL >= 33
   *out_usage = static_cast<uint64_t>(BufferUsage::FRONT_BUFFER);

   return 0;
#else
   return -ENOTSUP;
#endif
}

static int
destroy(struct u_gralloc *gralloc)
{
   gralloc_mapper *gr = (struct gralloc_mapper *)gralloc;
   if (gr->fallback)
      gr->fallback->ops.destroy(gr->fallback);
   delete gr;

   return 0;
}

struct u_gralloc *
u_gralloc_imapper_api_create()
{
   if (!ui.mapper && !libui_open())
      return NULL;
   auto &mapper = *ui.mapper;
   if(mapper.getMapperVersion() < GraphicBufferMapper::GRALLOC_4) {
      mesa_logi("Could not find IMapper v4/v5 API");
      return NULL;
   }
   auto gr = new gralloc_mapper;
   gr->fallback = u_gralloc_fallback_create();
   gr->base.ops.get_buffer_basic_info = mapper5_get_buffer_basic_info;
   gr->base.ops.get_buffer_color_info = mapper5_get_buffer_color_info;
   gr->base.ops.get_front_rendering_usage = mapper5_get_front_rendering_usage;
   gr->base.ops.destroy = destroy;

   /* XCLIPSE: the enum's numbering changed: Android 12 counts from GRALLOC_2 = 0 (the header
    * built against), Android 17 reports IMapper 5 as 5. Log the raw value. */
   mesa_logi("Using IMapper API through libui (mapper version enum %d)",
             (int)mapper.getMapperVersion());

   return &gr->base;
}
} // extern "C"

