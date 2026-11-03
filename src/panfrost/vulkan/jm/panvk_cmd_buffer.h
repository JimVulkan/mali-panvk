/*
 * Copyright © 2021 Collabora Ltd.
 * SPDX-License-Identifier: MIT
 */

#ifndef PANVK_CMD_BUFFER_H
#define PANVK_CMD_BUFFER_H

#ifndef PAN_ARCH
#error "PAN_ARCH must be defined"
#endif

#include <stdint.h>

#include "vulkan/runtime/vk_command_buffer.h"

#include "panvk_cmd_desc_state.h"
#include "panvk_cmd_dispatch.h"
#include "panvk_cmd_draw.h"
#include "panvk_cmd_push_constant.h"
#include "panvk_descriptor_set.h"
#include "panvk_descriptor_set_layout.h"
#include "panvk_device.h"
#include "panvk_macros.h"
#include "panvk_mempool.h"
#include "panvk_shader.h"

#include "pan_jc.h"

#include "util/list.h"

#include "genxml/gen_macros.h"

struct panvk_batch {
   struct list_head node;
   struct util_dynarray jobs;
   struct util_dynarray event_ops;
   struct pan_jc vtc_jc;
   struct pan_jc frag_jc;
   struct {
      struct pan_ptr desc;
      uint32_t desc_stride;
      uint32_t bo_count;

      /* One slot per color, two more slots for the depth/stencil buffers. */
      struct pan_kmod_bo *bos[MAX_RTS + 2];
      uint32_t layer_count;
   } fb;
   struct {
      struct pan_kmod_bo *src, *dst;
   } blit;
   struct pan_ptr tls;
   struct {
      struct pan_tiler_context ctx;
      struct pan_ptr heap_desc;
      struct pan_ptr ctx_descs;
      struct mali_tiler_heap_packed heap_templ;
      struct mali_tiler_context_packed ctx_templ;
   } tiler;
   struct pan_tls_info tlsinfo;
   unsigned wls_total_size;
   bool issued;
   /* Writes a GPU timestamp: kbase only runs the system timestamp for atoms with
    * BASE_JD_REQ_PERMON (a WRITE_VALUE SYSTEM_TIMESTAMP stores 0 without it). */
   bool needs_permon;
   /* WRITE_VALUE jobs in vtc_jc (panvk_emit_write_job). When they are all of it and there is no
    * fragment job, the atom can go to the fragment slot instead (gpu_queue). */
   uint16_t nr_write_jobs;
   /* kbase: the dev->tiler_heap its tiler context uses, and PANVK_BATCH_* ordering needs. */
   uint8_t tiler_heap;
   uint8_t order;
   /* Tessellation: this batch's struct poly_heap header, 0 until its first tessellated draw. */
   uint64_t tess_heap;
   /* Indirect draws, and indexed draws rerouted through them: one index scan and one patch job
    * serve all of the batch's draws. Both are queued at its first such draw (hdr.gpu != 0 from
    * then), the entries collect here, and the batch's close hands them to the GPU. */
   struct {
      struct pan_ptr hdr;
      unsigned ready_job;
      struct util_dynarray entries;
   } draw_helper;
};

/* On kbase, a batch's vertex/tiler atom normally waits only for the one before it and for the
 * fragment atom that last used its tiler heap, so it runs alongside the previous batch's
 * fragment atom (jm/panvk_vX_gpu_queue.c). These keep the two apart. */
/* Its vertex/tiler jobs depend on earlier fragment work: a barrier or event wait with a second
 * scope before the fragment stages, or device memory the fragment jobs may still use. */
#define PANVK_BATCH_WAIT_FRAG (1u << 0)
/* Its fragment jobs use device memory the next batch's vertex/tiler jobs may write. */
#define PANVK_BATCH_HOLD_FRAG (1u << 1)
/* Timestamp writes and timestamp pool resets (jm/panvk_vX_cmd_query.c): after all earlier work,
 * but in line with the fragment atoms, so the next batch's vertex/tiler atom does not wait for
 * them and still overlaps the previous fragment atom. A pending WAIT_FRAG passes on to the
 * batch after. */
#define PANVK_BATCH_FRAG_ORDERED (1u << 2)

/* The largest stack a command buffer keeps once per tiler heap (see cmd_close_batch). */
#define PANVK_KBASE_SHARED_STACK_MAX (32u * 1024 * 1024)

/* Pipeline stages only fragment jobs run. A dependency whose second scope lies within them is
 * kept by the fragment jobs' own order. */
#define PANVK_FRAG_ONLY_STAGES                                                                   \
   (VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |      \
    VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT |                                                \
    VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT |   \
    VK_PIPELINE_STAGE_2_FRAGMENT_SHADING_RATE_ATTACHMENT_BIT_KHR)

enum panvk_cmd_event_op_type {
   PANVK_EVENT_OP_SET,
   PANVK_EVENT_OP_RESET,
   PANVK_EVENT_OP_WAIT,
};

struct panvk_cmd_event_op {
   enum panvk_cmd_event_op_type type;
   struct panvk_event *event;
};

#if PAN_ARCH < 9
/* How many draws a tiler job is added to the chain behind: see
 * panvk_per_arch(cmd_flush_pending_tilers). */
#define PANVK_TILER_LAG 2

struct panvk_pending_tiler {
   struct pan_ptr job;
   unsigned dep;
};
#endif

struct panvk_cmd_buffer {
   struct vk_command_buffer vk;

   /* kbase: the last submission that ran this command buffer (0 if none). */
   uint64_t kbase_seq;

   struct panvk_pool desc_pool;
   struct panvk_pool varying_pool;
   struct panvk_pool tls_pool;
   struct list_head batches;
   struct list_head push_sets;
   struct panvk_batch *cur_batch;

   /* kbase: the stacks the batches of this command buffer share, the largest one asked for so
    * far, one per tiler heap. Two batches run at once only when their tiler heaps differ (a
    * fragment atom and the next batch's vertex/tiler atom), so they never share a stack. */
   struct {
      uint64_t ptr;
      unsigned size;
   } tls_stack[2];

   /* kbase: a barrier or event wait whose second scope reaches the vertex/tiler jobs was
    * recorded and no batch has taken it yet (PANVK_BATCH_WAIT_FRAG). Reset with the command
    * buffer; when still set at the end, the first batch after it in the queue takes it. */
   bool pending_frag_wait;
   /* kbase: the tiler heap of the next batch, plus one (0: take the device's turn). */
   uint8_t next_tiler_heap;

   struct {
      struct panvk_cmd_graphics_state gfx;
      struct panvk_cmd_compute_state compute;
      struct panvk_push_constant_state push_constants;
      bool uses_poly_heap;
      /* The struct poly_heap the indirect draws of this command buffer take their varyings
       * from, 0 until the first one. In the descriptor pool, not the device's: that one is
       * GPU-uncached on kbase, and the draw helper allocates from it with atomics. */
      uint64_t indirect_heap;
#if PAN_ARCH < 9
      /* Tiler jobs of the last draws, not in the chain yet. */
      struct panvk_pending_tiler pending_tilers[PANVK_TILER_LAG];
      unsigned pending_tiler_count;
#endif
#if PAN_ARCH == 9
      /* What the last draw built, reused by the next one while the state it came from has not
       * changed (jm/panvk_v9_cmd_draw.c). valid is cleared where that is not tracked. */
      struct panvk_cmd_buffer_v9_cache {
         bool valid;
         const void *fs;
         uint32_t base_instance;
         uint64_t blend;
         uint32_t blend_count;
         uint64_t zsd;
         uint64_t vs_fau;
         uint64_t fs_fau;
      } v9;
#endif
   } state;
};

VK_DEFINE_HANDLE_CASTS(panvk_cmd_buffer, vk.base, VkCommandBuffer,
                       VK_OBJECT_TYPE_COMMAND_BUFFER)

#define panvk_cmd_buffer_obj_list_init(cmdbuf, list_name)                      \
   list_inithead(&(cmdbuf)->list_name)

#define panvk_cmd_buffer_obj_list_cleanup(cmdbuf, list_name)                   \
   do {                                                                        \
      struct panvk_cmd_pool *__pool =                                          \
         container_of(cmdbuf->vk.pool, struct panvk_cmd_pool, vk);             \
      list_splicetail(&(cmdbuf)->list_name, &__pool->list_name);               \
   } while (0)

#define panvk_cmd_buffer_obj_list_reset(cmdbuf, list_name)                     \
   do {                                                                        \
      struct panvk_cmd_pool *__pool =                                          \
         container_of(cmdbuf->vk.pool, struct panvk_cmd_pool, vk);             \
      list_splicetail(&(cmdbuf)->list_name, &__pool->list_name);               \
      list_inithead(&(cmdbuf)->list_name);                                     \
   } while (0)

static inline struct panvk_descriptor_state *
panvk_cmd_get_desc_state(struct panvk_cmd_buffer *cmdbuf,
                         VkPipelineBindPoint bindpoint)
{
   switch (bindpoint) {
   case VK_PIPELINE_BIND_POINT_GRAPHICS:
      return &cmdbuf->state.gfx.desc_state;

   case VK_PIPELINE_BIND_POINT_COMPUTE:
      return &cmdbuf->state.compute.desc_state;

   default:
      assert(!"Unsupported bind point");
      return NULL;
   }
}

extern const struct vk_command_buffer_ops panvk_per_arch(cmd_buffer_ops);

struct panvk_batch *
   panvk_per_arch(cmd_open_batch)(struct panvk_cmd_buffer *cmdbuf);

void panvk_per_arch(cmd_close_batch)(struct panvk_cmd_buffer *cmdbuf);

#if PAN_ARCH < 9
void panvk_per_arch(cmd_flush_pending_tilers)(struct panvk_cmd_buffer *cmdbuf,
                                              unsigned keep);
#endif

void panvk_per_arch(cmd_finish_draw_helper)(struct panvk_cmd_buffer *cmdbuf,
                                            struct panvk_batch *batch);

VkResult panvk_per_arch(cmd_alloc_fb_desc)(struct panvk_cmd_buffer *cmdbuf);

VkResult panvk_per_arch(cmd_alloc_tls_desc)(struct panvk_cmd_buffer *cmdbuf,
                                            bool gfx);

VkResult
   panvk_per_arch(cmd_prepare_tiler_context)(struct panvk_cmd_buffer *cmdbuf,
                                             uint32_t layer_idx);

void panvk_per_arch(cmd_bind_shaders)(struct vk_command_buffer *vk_cmd,
                                      uint32_t stage_count,
                                      const mesa_shader_stage *stages,
                                      struct vk_shader **const shaders);

/* If a job was added to reset the heap, it's job id is returned in 'job_id'
 * so that it can be used in dependencies */
VkResult panvk_per_arch(cmd_init_poly_heap)(struct panvk_cmd_buffer *cmdbuf,
                                            unsigned *job_id);

#endif
