/*
 * Copyright 2026 JimVulkan
 * SPDX-License-Identifier: MIT
 */

/*
 * Draws on Valhall's first generation (v9: Mali-G57, G77, G68, G78), which has Bifrost's job
 * manager with Valhall's shaders and descriptors. A draw is one MALLOC_VERTEX job: the
 * position shader, the varying shader and the tiler in one, the varyings in memory the GPU
 * allocates itself. The descriptors are the ones the CSF path builds for v10
 * (csf/panvk_vX_cmd_draw.c); here they go into the job's sections instead of CS registers.
 *
 * Geometry and tessellation shaders, polygon modes and transform feedback run through the
 * emulation (gs_draw, tess_draw) as COMPUTE jobs ahead of the rasterization job.
 */

#include "genxml/gen_macros.h"

#include "panvk_buffer.h"
#include "panvk_cmd_alloc.h"
#include "panvk_cmd_buffer.h"
#include "panvk_cmd_desc_state.h"
#include "panvk_cmd_draw.h"
#include "panvk_cmd_meta.h"
#include "panvk_cmd_precomp.h"
#include "panvk_device.h"
#include "panvk_entrypoints.h"
#include "panvk_image.h"
#include "panvk_image_view.h"
#include "panvk_instance.h"
#include "panvk_priv_bo.h"
#include "panvk_gs.h"
#include "panvk_shader.h"

#include "pan_desc.h"
#include "pan_earlyzs.h"
#include "pan_encoder.h"
#include "pan_format.h"
#include "pan_jc.h"
#include "pan_props.h"
#include "pan_shader.h"

#include "draw_helper.h"
#include "libpan_dgc.h"
#include "libpan_shaders.h"

#include "poly/geometry.h"
#include "poly/tessellator.h"
#include "util/log.h"
#include "vk_format.h"
#include "vk_meta.h"

static bool
has_depth_att(struct panvk_cmd_buffer *cmdbuf)
{
   return (cmdbuf->state.gfx.render.bound_attachments &
           MESA_VK_RP_ATTACHMENT_DEPTH_BIT) != 0;
}

static bool
has_stencil_att(struct panvk_cmd_buffer *cmdbuf)
{
   return (cmdbuf->state.gfx.render.bound_attachments &
           MESA_VK_RP_ATTACHMENT_STENCIL_BIT) != 0;
}

static bool
writes_depth(struct panvk_cmd_buffer *cmdbuf)
{
   const struct vk_depth_stencil_state *ds =
      &cmdbuf->vk.dynamic_graphics_state.ds;

   return has_depth_att(cmdbuf) && ds->depth.test_enable &&
          ds->depth.write_enable && ds->depth.compare_op != VK_COMPARE_OP_NEVER;
}

static bool
writes_stencil(struct panvk_cmd_buffer *cmdbuf)
{
   const struct vk_depth_stencil_state *ds =
      &cmdbuf->vk.dynamic_graphics_state.ds;

   return has_stencil_att(cmdbuf) && ds->stencil.test_enable &&
          ((ds->stencil.front.write_mask &&
            (ds->stencil.front.op.fail != VK_STENCIL_OP_KEEP ||
             ds->stencil.front.op.pass != VK_STENCIL_OP_KEEP ||
             ds->stencil.front.op.depth_fail != VK_STENCIL_OP_KEEP)) ||
           (ds->stencil.back.write_mask &&
            (ds->stencil.back.op.fail != VK_STENCIL_OP_KEEP ||
             ds->stencil.back.op.pass != VK_STENCIL_OP_KEEP ||
             ds->stencil.back.op.depth_fail != VK_STENCIL_OP_KEEP)));
}

static bool
ds_test_always_passes(struct panvk_cmd_buffer *cmdbuf)
{
   const struct vk_depth_stencil_state *ds =
      &cmdbuf->vk.dynamic_graphics_state.ds;

   if (!has_depth_att(cmdbuf))
      return true;

   if (ds->depth.test_enable && ds->depth.compare_op != VK_COMPARE_OP_ALWAYS)
      return false;

   if (ds->stencil.test_enable &&
       (ds->stencil.front.op.compare != VK_COMPARE_OP_ALWAYS ||
        ds->stencil.back.op.compare != VK_COMPARE_OP_ALWAYS))
      return false;

   return true;
}

static inline enum mali_func
translate_compare_func(VkCompareOp comp)
{
   STATIC_ASSERT(VK_COMPARE_OP_NEVER == (VkCompareOp)MALI_FUNC_NEVER);
   STATIC_ASSERT(VK_COMPARE_OP_ALWAYS == (VkCompareOp)MALI_FUNC_ALWAYS);
   return (enum mali_func)comp;
}

static enum mali_stencil_op
translate_stencil_op(VkStencilOp in)
{
   switch (in) {
   case VK_STENCIL_OP_KEEP:
      return MALI_STENCIL_OP_KEEP;
   case VK_STENCIL_OP_ZERO:
      return MALI_STENCIL_OP_ZERO;
   case VK_STENCIL_OP_REPLACE:
      return MALI_STENCIL_OP_REPLACE;
   case VK_STENCIL_OP_INCREMENT_AND_CLAMP:
      return MALI_STENCIL_OP_INCR_SAT;
   case VK_STENCIL_OP_DECREMENT_AND_CLAMP:
      return MALI_STENCIL_OP_DECR_SAT;
   case VK_STENCIL_OP_INCREMENT_AND_WRAP:
      return MALI_STENCIL_OP_INCR_WRAP;
   case VK_STENCIL_OP_DECREMENT_AND_WRAP:
      return MALI_STENCIL_OP_DECR_WRAP;
   case VK_STENCIL_OP_INVERT:
      return MALI_STENCIL_OP_INVERT;
   default:
      UNREACHABLE("Invalid stencil op");
   }
}

static enum mali_draw_mode
translate_prim(enum mesa_prim prim)
{
   switch (prim) {
   case MESA_PRIM_POINTS:
      return MALI_DRAW_MODE_POINTS;
   case MESA_PRIM_LINES:
      return MALI_DRAW_MODE_LINES;
   case MESA_PRIM_LINE_STRIP:
      return MALI_DRAW_MODE_LINE_STRIP;
   case MESA_PRIM_TRIANGLES:
      return MALI_DRAW_MODE_TRIANGLES;
   case MESA_PRIM_TRIANGLE_STRIP:
      return MALI_DRAW_MODE_TRIANGLE_STRIP;
   case MESA_PRIM_TRIANGLE_FAN:
      return MALI_DRAW_MODE_TRIANGLE_FAN;
   case MESA_PRIM_LINES_ADJACENCY:
      return MALI_DRAW_MODE_LINES_ADJACENCY;
   case MESA_PRIM_LINE_STRIP_ADJACENCY:
      return MALI_DRAW_MODE_LINE_STRIP_ADJACENCY;
   case MESA_PRIM_TRIANGLES_ADJACENCY:
      return MALI_DRAW_MODE_TRIANGLES_ADJACENCY;
   case MESA_PRIM_TRIANGLE_STRIP_ADJACENCY:
      return MALI_DRAW_MODE_TRIANGLE_STRIP_ADJACENCY;
   default:
      UNREACHABLE("Invalid primitive type");
   }
}

static enum mali_index_type
index_size_to_index_type(uint32_t size)
{
   switch (size) {
   case 0:
      return MALI_INDEX_TYPE_NONE;
   case 1:
      return MALI_INDEX_TYPE_UINT8;
   case 2:
      return MALI_INDEX_TYPE_UINT16;
   case 4:
      return MALI_INDEX_TYPE_UINT32;
   default:
      assert(!"Invalid index size");
      return MALI_INDEX_TYPE_NONE;
   }
}

static void
set_provoking_vertex_mode(struct panvk_cmd_buffer *cmdbuf,
                          enum u_tristate first_provoking_vertex)
{
   struct panvk_cmd_graphics_state *state = &cmdbuf->state.gfx;

   if (first_provoking_vertex != U_TRISTATE_UNSET)
      state->render.first_provoking_vertex = first_provoking_vertex;

   /* The FBD is emitted when the batch closes, with the last mode asked for. */
   if (state->render.first_provoking_vertex == U_TRISTATE_UNSET)
      state->render.first_provoking_vertex = U_TRISTATE_YES;
}

/* --- descriptors (as csf/panvk_vX_cmd_draw.c) ------------------------------------------------ */

static void
emit_vs_attrib(struct panvk_cmd_buffer *cmdbuf, uint32_t attrib_idx,
               uint32_t vb_desc_offset, uint32_t base_instance,
               struct mali_attribute_packed *desc)
{
   const struct vk_dynamic_graphics_state *dyns =
      &cmdbuf->vk.dynamic_graphics_state;
   const struct vk_vertex_input_state *vi = dyns->vi;
   const struct vk_vertex_attribute_state *attrib_info =
      &vi->attributes[attrib_idx];
   const struct vk_vertex_binding_state *buf_info =
      &vi->bindings[attrib_info->binding];
   const uint32_t stride = dyns->vi_binding_strides[attrib_info->binding];
   bool per_instance = buf_info->input_rate == VK_VERTEX_INPUT_RATE_INSTANCE;
   enum pipe_format f = vk_format_to_pipe_format(attrib_info->format);
   unsigned buf_idx = vb_desc_offset + attrib_info->binding;

   pan_pack(desc, ATTRIBUTE, cfg) {
      cfg.offset = attrib_info->offset;

      if (per_instance)
         cfg.offset += base_instance * stride;

      cfg.format = GENX(pan_format_from_pipe_format)(f)->hw;
      cfg.table = 0;
      cfg.buffer_index = buf_idx;
      cfg.stride = stride;
      if (!per_instance) {
         cfg.attribute_type = MALI_ATTRIBUTE_TYPE_1D;
         cfg.frequency = MALI_ATTRIBUTE_FREQUENCY_VERTEX;
         cfg.offset_enable = true;
      } else if (buf_info->divisor == 1) {
         cfg.attribute_type = MALI_ATTRIBUTE_TYPE_1D;
         cfg.frequency = MALI_ATTRIBUTE_FREQUENCY_INSTANCE;
      } else if (buf_info->divisor == 0) {
         cfg.attribute_type = MALI_ATTRIBUTE_TYPE_1D;
         cfg.frequency = MALI_ATTRIBUTE_FREQUENCY_INSTANCE;
         cfg.stride = 0;
      } else if (util_is_power_of_two_or_zero(buf_info->divisor)) {
         cfg.attribute_type = MALI_ATTRIBUTE_TYPE_1D_POT_DIVISOR;
         cfg.frequency = MALI_ATTRIBUTE_FREQUENCY_INSTANCE;
         cfg.divisor_r = __builtin_ctz(buf_info->divisor);
      } else {
         cfg.attribute_type = MALI_ATTRIBUTE_TYPE_1D_NPOT_DIVISOR;
         cfg.frequency = MALI_ATTRIBUTE_FREQUENCY_INSTANCE;
         cfg.divisor_d = pan_compute_npot_divisor(
            buf_info->divisor, &cfg.divisor_r, &cfg.divisor_e);
      }
   }
}

static VkResult
prepare_vs_driver_set(struct panvk_cmd_buffer *cmdbuf, uint32_t base_instance)
{
   const struct panvk_shader_desc_info *vs_desc_info =
      &cmdbuf->state.gfx.vs.shader->desc_info;
   struct panvk_shader_desc_state *vs_desc_state = &cmdbuf->state.gfx.vs.desc;
   const struct vk_dynamic_graphics_state *dyns =
      &cmdbuf->vk.dynamic_graphics_state;
   const struct vk_vertex_input_state *vi = dyns->vi;

   uint32_t vb_count = 0;
   u_foreach_bit(i, vi->attributes_valid)
      vb_count = MAX2(vi->attributes[i].binding + 1, vb_count);

   uint32_t vb_offset = vs_desc_info->dyn_bufs.count + MAX_VS_ATTRIBS + 1;
   uint32_t desc_count = vb_offset + vb_count;

   const struct panvk_descriptor_state *desc_state =
      &cmdbuf->state.gfx.desc_state;
   struct pan_ptr driver_set = panvk_cmd_alloc_dev_mem(
      cmdbuf, desc, desc_count * PANVK_DESCRIPTOR_SIZE, PANVK_DESCRIPTOR_SIZE);
   struct panvk_opaque_desc *descs = driver_set.cpu;

   if (!driver_set.gpu)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;

   for (uint32_t i = 0; i < MAX_VS_ATTRIBS; i++) {
      if (vi->attributes_valid & BITFIELD_BIT(i)) {
         emit_vs_attrib(cmdbuf, i, vb_offset, base_instance,
                        (struct mali_attribute_packed *)(&descs[i]));
      } else {
         pan_cast_and_pack(&descs[i], NULL_DESCRIPTOR, cfg)
            ;
      }
   }

   /* Dummy sampler always comes right after the vertex attribs. */
   pan_cast_and_pack(&descs[MAX_VS_ATTRIBS], SAMPLER, cfg) {
      cfg.clamp_integer_array_indices = false;
   }

   panvk_per_arch(cmd_fill_dyn_bufs)(
      desc_state, vs_desc_info,
      (struct mali_buffer_packed *)(&descs[MAX_VS_ATTRIBS + 1]));

   for (uint32_t i = 0; i < vb_count; i++) {
      const struct panvk_attrib_buf *vb = &cmdbuf->state.gfx.vb.bufs[i];
      const bool nulldesc = (vb->address == 0 && vb->size == 0);

      if ((vi->bindings_valid & BITFIELD_BIT(i)) && !nulldesc) {
         pan_cast_and_pack(&descs[vb_offset + i], BUFFER, cfg) {
            cfg.address = vb->address;
            cfg.size = vb->size;
         }
      } else {
         pan_cast_and_pack(&descs[vb_offset + i], NULL_DESCRIPTOR, cfg)
            ;
      }
   }

   vs_desc_state->driver_set.dev_addr = driver_set.gpu;
   vs_desc_state->driver_set.size = desc_count * PANVK_DESCRIPTOR_SIZE;
   return VK_SUCCESS;
}

static void
emit_varying_descs(const struct panvk_cmd_buffer *cmdbuf,
                   struct mali_attribute_packed *descs)
{
   const struct panvk_shader_variant *vs =
      panvk_shader_hw_variant(cmdbuf->state.gfx.vs.shader);
   const struct panvk_shader_variant *fs =
      panvk_shader_only_variant(get_fs(cmdbuf));

   const struct pan_varying_layout *vs_layout = &vs->info.varyings.formats;
   const struct pan_varying_layout *fs_format = &fs->info.varyings.formats;
   pan_varying_layout_require_layout(vs_layout);
   pan_varying_layout_require_format(fs_format);

   for (uint32_t i = 0; i < fs_format->count; i++) {
      const struct pan_varying_slot *fs_slot =
         pan_varying_layout_slot_at(fs_format, i);

      if (!fs_slot || fs_slot->section != PAN_VARYING_SECTION_GENERIC)
         continue;

      unsigned offset = 0;
      enum pipe_format format = PIPE_FORMAT_NONE;

      const struct pan_varying_slot *vs_slot =
         pan_varying_layout_find_slot(vs_layout, fs_slot->location);
      if (vs_slot) {
         nir_alu_type base_type = nir_alu_type_get_base_type(fs_slot->alu_type);
         nir_alu_type bit_size = nir_alu_type_get_type_size(vs_slot->alu_type);

         offset = vs_slot->offset;
         format = pan_varying_format(base_type | bit_size, vs_slot->ncomps);
      }

      pan_pack(&descs[i], ATTRIBUTE, cfg) {
         cfg.attribute_type = MALI_ATTRIBUTE_TYPE_VERTEX_PACKET;
         cfg.offset_enable = false;
         cfg.format = GENX(pan_format_from_pipe_format)(format)->hw;
         cfg.table = 61;
         cfg.frequency = MALI_ATTRIBUTE_FREQUENCY_VERTEX;
         cfg.offset = 1024 + offset;
         cfg.buffer_index = 0;
         cfg.attribute_stride = vs_layout->generic_size_B;
         cfg.packet_stride = vs_layout->generic_size_B + 16;
      }
   }
}

static VkResult
prepare_fs_driver_set(struct panvk_cmd_buffer *cmdbuf)
{
   const struct panvk_shader_desc_info *fs_desc_info =
      &cmdbuf->state.gfx.fs.shader->desc_info;
   struct panvk_shader_desc_state *fs_desc_state = &cmdbuf->state.gfx.fs.desc;
   const struct panvk_descriptor_state *desc_state =
      &cmdbuf->state.gfx.desc_state;
   const uint32_t desc_count = fs_desc_info->fs_varying_attr_desc_count +
                               fs_desc_info->dyn_bufs.count + 1;
   struct pan_ptr driver_set = panvk_cmd_alloc_dev_mem(
      cmdbuf, desc, desc_count * PANVK_DESCRIPTOR_SIZE, PANVK_DESCRIPTOR_SIZE);
   struct panvk_opaque_desc *descs = driver_set.cpu;

   if (desc_count && !driver_set.gpu)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;

   if (fs_desc_info->fs_varying_attr_desc_count > 0)
      emit_varying_descs(cmdbuf, (struct mali_attribute_packed *)(&descs[0]));

   const uint32_t sampler_idx = fs_desc_info->fs_varying_attr_desc_count;
   pan_cast_and_pack(&descs[sampler_idx], SAMPLER, cfg) {
      cfg.clamp_integer_array_indices = false;
   }

   panvk_per_arch(cmd_fill_dyn_bufs)(
      desc_state, fs_desc_info,
      (struct mali_buffer_packed *)(&descs[sampler_idx + 1]));

   fs_desc_state->driver_set.dev_addr = driver_set.gpu;
   fs_desc_state->driver_set.size = desc_count * PANVK_DESCRIPTOR_SIZE;
   return VK_SUCCESS;
}

/* The tables are rebuilt only when what they come from changed: a draw that rebuilds them all
 * costs more CPU than the vendor driver spends on it, and the GPU fetches them again too. */
static bool
vs_desc_dirty(struct panvk_cmd_buffer *cmdbuf, const struct panvk_draw_info *draw)
{
   return !cmdbuf->state.v9.valid ||
          cmdbuf->state.v9.base_instance != draw->instance.base ||
          dyn_gfx_state_dirty(cmdbuf, VI) ||
          dyn_gfx_state_dirty(cmdbuf, VI_BINDINGS_VALID) ||
          dyn_gfx_state_dirty(cmdbuf, VI_BINDING_STRIDES) ||
          gfx_state_dirty(cmdbuf, VB) || gfx_state_dirty(cmdbuf, VS) ||
          gfx_state_dirty(cmdbuf, DESC_STATE);
}

static bool
fs_desc_dirty(struct panvk_cmd_buffer *cmdbuf)
{
   return !cmdbuf->state.v9.valid || fs_user_dirty(cmdbuf) ||
          gfx_state_dirty(cmdbuf, VS) || gfx_state_dirty(cmdbuf, DESC_STATE);
}

static VkResult
prepare_descs(struct panvk_cmd_buffer *cmdbuf, const struct panvk_draw_info *draw)
{
   const struct panvk_shader *vs = cmdbuf->state.gfx.vs.shader;
   const struct panvk_shader *fs = get_fs(cmdbuf);
   struct panvk_descriptor_state *desc_state = &cmdbuf->state.gfx.desc_state;
   VkResult result;

   uint32_t used_set_mask = vs->desc_info.used_set_mask;
   used_set_mask |= fs ? fs->desc_info.used_set_mask : 0;

   const bool vs_dirty = vs_desc_dirty(cmdbuf, draw);
   const bool fs_dirty = fs_desc_dirty(cmdbuf);
   if (!vs_dirty && !fs_dirty)
      return VK_SUCCESS;

   result = panvk_per_arch(cmd_prepare_push_descs)(cmdbuf, desc_state,
                                                   used_set_mask);
   if (result != VK_SUCCESS)
      return result;

   if (vs_dirty) {
      result = prepare_vs_driver_set(cmdbuf, draw->instance.base);
      if (result != VK_SUCCESS)
         return result;

      result = panvk_per_arch(cmd_prepare_shader_res_table)(
         cmdbuf, desc_state, &vs->desc_info, &cmdbuf->state.gfx.vs.desc, 1);
      if (result != VK_SUCCESS)
         return result;
      cmdbuf->state.v9.base_instance = draw->instance.base;
   }

   struct panvk_shader_desc_state *fs_desc_state = &cmdbuf->state.gfx.fs.desc;
   if (!fs) {
      memset(fs_desc_state, 0, sizeof(*fs_desc_state));
      return VK_SUCCESS;
   }
   if (!fs_dirty)
      return VK_SUCCESS;

   result = prepare_fs_driver_set(cmdbuf);
   if (result != VK_SUCCESS)
      return result;

   return panvk_per_arch(cmd_prepare_shader_res_table)(
      cmdbuf, desc_state, &fs->desc_info, fs_desc_state, 1);
}

static VkResult
build_blend(struct panvk_cmd_buffer *cmdbuf,
            const struct panvk_shader_variant *fs, uint64_t *bds_gpu,
            uint32_t *bd_count_out)
{
   uint32_t bd_count = cmdbuf->state.gfx.render.fb.layout.rt_count;
   struct pan_ptr ptr = panvk_cmd_alloc_desc_array(cmdbuf, bd_count, BLEND);
   struct mali_blend_packed *bds = ptr.cpu;

   if (bd_count && !ptr.gpu)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;

   if (fs) {
      VkResult result = panvk_per_arch(blend_emit_descs)(cmdbuf, bds);
      if (result != VK_SUCCESS)
         return result;
   } else {
      for (unsigned i = 0; i < bd_count; i++) {
         pan_pack(&bds[i], BLEND, cfg) {
            cfg.enable = false;
            cfg.internal.mode = MALI_BLEND_MODE_OFF;
         }
      }
   }

   *bds_gpu = ptr.gpu;
   *bd_count_out = bd_count;
   return VK_SUCCESS;
}

static VkResult
build_zsd(struct panvk_cmd_buffer *cmdbuf, uint64_t *zsd_gpu)
{
   const struct vk_dynamic_graphics_state *dyns =
      &cmdbuf->vk.dynamic_graphics_state;
   const struct vk_depth_stencil_state *ds = &dyns->ds;
   const struct vk_rasterization_state *rs = &dyns->rs;
   bool test_s = has_stencil_att(cmdbuf) && ds->stencil.test_enable;
   bool test_z = has_depth_att(cmdbuf) && ds->depth.test_enable;
   const struct panvk_shader_variant *fs =
      panvk_shader_only_variant(get_fs(cmdbuf));

   struct pan_ptr zsd = panvk_cmd_alloc_desc(cmdbuf, DEPTH_STENCIL);
   if (!zsd.gpu)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;

   pan_cast_and_pack(zsd.cpu, DEPTH_STENCIL, cfg) {
      cfg.stencil_test_enable = test_s;
      if (test_s) {
         cfg.front_compare_function =
            translate_compare_func(ds->stencil.front.op.compare);
         cfg.front_stencil_fail =
            translate_stencil_op(ds->stencil.front.op.fail);
         cfg.front_depth_fail =
            translate_stencil_op(ds->stencil.front.op.depth_fail);
         cfg.front_depth_pass = translate_stencil_op(ds->stencil.front.op.pass);
         cfg.back_compare_function =
            translate_compare_func(ds->stencil.back.op.compare);
         cfg.back_stencil_fail = translate_stencil_op(ds->stencil.back.op.fail);
         cfg.back_depth_fail =
            translate_stencil_op(ds->stencil.back.op.depth_fail);
         cfg.back_depth_pass = translate_stencil_op(ds->stencil.back.op.pass);
      }

      cfg.stencil_from_shader = fs ? fs->info.fs.writes_stencil : 0;
      cfg.front_write_mask = ds->stencil.front.write_mask;
      cfg.back_write_mask = ds->stencil.back.write_mask;
      cfg.front_value_mask = ds->stencil.front.compare_mask;
      cfg.back_value_mask = ds->stencil.back.compare_mask;
      cfg.front_reference_value = ds->stencil.front.reference;
      cfg.back_reference_value = ds->stencil.back.reference;

      cfg.depth_cull_enable = vk_rasterization_state_depth_clip_enable(rs);
      if (rs->depth_clamp_enable)
         cfg.depth_clamp_mode = MALI_DEPTH_CLAMP_MODE_BOUNDS;

      if (fs)
         cfg.depth_source = pan_depth_source(&fs->info);

      cfg.depth_write_enable = test_z && ds->depth.write_enable;
      cfg.depth_bias_enable = rs->depth_bias.enable;
      cfg.depth_function = test_z ? translate_compare_func(ds->depth.compare_op)
                                  : MALI_FUNC_ALWAYS;
      cfg.depth_units = rs->depth_bias.constant_factor;
      cfg.depth_factor = rs->depth_bias.slope_factor;
      cfg.depth_bias_clamp = rs->depth_bias.clamp;
   }

   *zsd_gpu = zsd.gpu;
   return VK_SUCCESS;
}

/* DCD flags 0 and 1; v9 has no flags 2, its read/write masks come from the blend descriptors. */
static void
build_dcd_flags(struct panvk_cmd_buffer *cmdbuf,
                const struct panvk_shader_variant *fs, enum mesa_prim prim,
                struct mali_dcd_flags_0_packed *flags_0,
                struct mali_dcd_flags_1_packed *flags_1)
{
   const struct vk_dynamic_graphics_state *dyns =
      &cmdbuf->vk.dynamic_graphics_state;
   const struct vk_rasterization_state *rs = &dyns->rs;

   bool alpha_to_coverage = dyns->ms.alpha_to_coverage_enable;
   bool writes_z = writes_depth(cmdbuf);
   bool writes_s = writes_stencil(cmdbuf);
   uint8_t rt_mask = cmdbuf->state.gfx.render.bound_attachments &
                     MESA_VK_RP_ATTACHMENT_ANY_COLOR_BITS;
   uint8_t rt_written = 0, rt_read = 0;
   bool shader_modifies_coverage = false;

   if (fs) {
      rt_written = color_attachment_written_mask(fs, &dyns->cal);
      rt_read = color_attachment_read_mask(fs, &dyns->ial, rt_mask);
      shader_modifies_coverage = fs->info.fs.writes_coverage ||
                                 fs->info.fs.can_discard || alpha_to_coverage;
   }

   bool msaa = dyns->ms.rasterization_samples > 1;
   enum mesa_prim reduced_prim = u_reduced_prim(prim);
   if (reduced_prim == MESA_PRIM_LINES &&
       rs->line.mode == VK_LINE_RASTERIZATION_MODE_BRESENHAM)
      msaa = false;

   pan_pack(flags_0, DCD_FLAGS_0, cfg) {
      if (fs) {
         enum pan_earlyzs_zs_tilebuf_read zs_read =
            PAN_EARLYZS_ZS_TILEBUF_NOT_READ;

         if (z_attachment_read(fs, &dyns->ial) ||
             s_attachment_read(fs, &dyns->ial))
            zs_read = PAN_EARLYZS_ZS_TILEBUF_READ_NO_OPT;

         bool roa_color = dyns->rasterization_order_access &
                          VK_IMAGE_ASPECT_COLOR_BIT;

         cfg.allow_forward_pixel_to_kill =
            fs->info.fs.can_fpk && !(rt_mask & ~rt_written) &&
            !(rt_read & rt_written) && !alpha_to_coverage &&
            !cmdbuf->state.gfx.cb.info.any_dest_read && !roa_color;

         cfg.allow_forward_pixel_to_be_killed = !fs->info.writes_global;

         bool oq = cmdbuf->state.gfx.occlusion_query.mode !=
                   MALI_OCCLUSION_MODE_DISABLED;
         if (dyns->rasterization_order_access &
             (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT))
            zs_read = PAN_EARLYZS_ZS_TILEBUF_READ_NO_OPT;

         struct pan_earlyzs_state earlyzs =
            pan_earlyzs_get(fs->fs.earlyzs_lut, writes_z || writes_s || oq,
                            alpha_to_coverage, ds_test_always_passes(cmdbuf),
                            zs_read);

         cfg.pixel_kill_operation = (enum mali_pixel_kill)earlyzs.kill;
         cfg.zs_update_operation = (enum mali_pixel_kill)earlyzs.update;
         cfg.evaluate_per_sample = (fs->info.fs.sample_shading ||
                                    cmdbuf->state.gfx.cb.info.needs_shader) &&
                                   (dyns->ms.rasterization_samples > 1);
         cfg.shader_modifies_coverage = shader_modifies_coverage;
      } else {
         cfg.allow_forward_pixel_to_kill = true;
         cfg.allow_forward_pixel_to_be_killed = true;
         cfg.pixel_kill_operation = MALI_PIXEL_KILL_FORCE_EARLY;
         cfg.zs_update_operation = MALI_PIXEL_KILL_FORCE_EARLY;
         cfg.overdraw_alpha0 = true;
         cfg.overdraw_alpha1 = true;
      }

      if (rs->line.mode == VK_LINE_RASTERIZATION_MODE_BRESENHAM)
         cfg.aligned_line_ends = true;

      cfg.front_face_ccw = rs->front_face == VK_FRONT_FACE_COUNTER_CLOCKWISE;

      const bool non_polygon = reduced_prim != MESA_PRIM_TRIANGLES;
      cfg.cull_front_face =
         !non_polygon && (rs->cull_mode & VK_CULL_MODE_FRONT_BIT) != 0;
      cfg.cull_back_face =
         !non_polygon && (rs->cull_mode & VK_CULL_MODE_BACK_BIT) != 0;

      cfg.multisample_enable = msaa;
      cfg.occlusion_query = cmdbuf->state.gfx.occlusion_query.mode;
      cfg.alpha_to_coverage = alpha_to_coverage;
      cfg.scissor_to_bounding_box = true;
   }

   pan_pack(flags_1, DCD_FLAGS_1, cfg) {
      cfg.sample_mask = dyns->ms.sample_mask;
      cfg.render_target_mask = rt_written;
   }
}

/* The scissor box is the viewport clipped by the scissor, as on v10. */
static void
emit_scissor(struct panvk_cmd_buffer *cmdbuf, void *out)
{
   pan_pack((struct mali_scissor_packed *)out, SCISSOR, cfg) {
      int minx, miny, maxx, maxy;
      panvk_viewport_box(&cmdbuf->state.gfx, &cmdbuf->vk.dynamic_graphics_state.vp, true, &minx,
                         &miny, &maxx, &maxy);

      maxx = maxx > minx ? maxx - 1 : maxx;
      maxy = maxy > miny ? maxy - 1 : maxy;

      cfg.scissor_minimum_x = CLAMP(minx, 0, UINT16_MAX);
      cfg.scissor_minimum_y = CLAMP(miny, 0, UINT16_MAX);
      cfg.scissor_maximum_x = CLAMP(maxx, 0, UINT16_MAX);
      cfg.scissor_maximum_y = CLAMP(maxy, 0, UINT16_MAX);
   }
}

static void
emit_primitive_size(struct panvk_cmd_buffer *cmdbuf,
                    const struct panvk_shader_variant *vs,
                    enum mesa_prim prim, void *out)
{
   float size = 1.0f;

   switch (u_reduced_prim(prim)) {
   case MESA_PRIM_POINTS:
      size = 1.0f;
      break;
   case MESA_PRIM_LINES:
      size = cmdbuf->vk.dynamic_graphics_state.rs.line.width;
      break;
   default:
      break;
   }

   pan_pack((struct mali_primitive_size_packed *)out, PRIMITIVE_SIZE, cfg) {
      cfg.fixed_sized = size;
   }
}

static void
pack_shader_env(struct MALI_SHADER_ENVIRONMENT *cfg, uint64_t res_table,
                uint64_t spd, uint64_t tls, uint64_t fau, uint32_t fau_count)
{
   cfg->resources = res_table;
   cfg->shader = spd;
   cfg->thread_storage = tls;
   cfg->fau = fau;
   cfg->fau_count = fau_count;
}

/* --- draws ----------------------------------------------------------------------------------- */

struct v9_draw_state {
   /* The vertex shader of the MALLOC_VERTEX job: the application's, or with a geometry shader
    * the variant that reads the emulation's output back. */
   const struct panvk_shader_variant *vs;
   uint64_t vs_res;
   uint64_t vs_fau;
   uint64_t blend;
   uint32_t blend_count;
   uint64_t zsd;
   struct mali_dcd_flags_0_packed flags_0;
   struct mali_dcd_flags_1_packed flags_1;
   float min_z, max_z;
};

static VkResult
prepare_draw(struct panvk_cmd_buffer *cmdbuf, const struct panvk_draw_info *draw,
             struct v9_draw_state *st)
{
   struct panvk_batch *batch = cmdbuf->cur_batch;
   const struct panvk_shader_variant *vs =
      panvk_shader_hw_variant(cmdbuf->state.gfx.vs.shader);
   const struct panvk_shader_variant *fs =
      panvk_shader_only_variant(get_fs(cmdbuf));
   VkResult result;

   /* A MALLOC_VERTEX job per layer, and 16 bits of job index. */
   if (batch->vtc_jc.job_index + (2 * cmdbuf->state.gfx.render.layer_count) >=
       UINT16_MAX) {
      panvk_per_arch(cmd_close_batch)(cmdbuf);
      batch = panvk_per_arch(cmd_open_batch)(cmdbuf);
   }

   if (cmdbuf->state.gfx.vk_meta) {
      set_provoking_vertex_mode(cmdbuf, U_TRISTATE_UNSET);
   } else {
      set_provoking_vertex_mode(
         cmdbuf, u_tristate_make(cmdbuf->vk.dynamic_graphics_state.rs.provoking_vertex ==
                                 VK_PROVOKING_VERTEX_MODE_FIRST_VERTEX_EXT));
   }

   if (!cmdbuf->vk.dynamic_graphics_state.rs.rasterizer_discard_enable) {
      uint32_t *nr_samples = &cmdbuf->state.gfx.render.fb.nr_samples;
      if (!batch->fb.desc.gpu && !cmdbuf->state.gfx.render.bound_attachments)
         *nr_samples = cmdbuf->vk.dynamic_graphics_state.ms.rasterization_samples;

      result = panvk_per_arch(cmd_alloc_fb_desc)(cmdbuf);
      if (result != VK_SUCCESS)
         return result;
   }

   panvk_per_arch(cmd_select_tile_size)(cmdbuf);

   result = panvk_per_arch(cmd_alloc_tls_desc)(cmdbuf, true);
   if (result != VK_SUCCESS)
      return result;

   batch->tlsinfo.tls.size = MAX3(vs->info.tls_size, fs ? fs->info.tls_size : 0,
                                  batch->tlsinfo.tls.size);

   /* Whether the fragment shader runs changes the tables, the blend and the depth/stencil. */
   if (cmdbuf->state.v9.fs != fs)
      cmdbuf->state.v9.valid = false;

   result = prepare_descs(cmdbuf, draw);
   if (result != VK_SUCCESS)
      return result;

   struct panvk_cmd_buffer_v9_cache *c = &cmdbuf->state.v9;
   const bool blend_dirty =
      !c->valid || fs_user_dirty(cmdbuf) ||
      dyn_gfx_state_dirty(cmdbuf, MS_ALPHA_TO_ONE_ENABLE) ||
      dyn_gfx_state_dirty(cmdbuf, CB_LOGIC_OP_ENABLE) ||
      dyn_gfx_state_dirty(cmdbuf, CB_LOGIC_OP) ||
      dyn_gfx_state_dirty(cmdbuf, CB_ATTACHMENT_COUNT) ||
      dyn_gfx_state_dirty(cmdbuf, CB_COLOR_WRITE_ENABLES) ||
      dyn_gfx_state_dirty(cmdbuf, CB_BLEND_ENABLES) ||
      dyn_gfx_state_dirty(cmdbuf, CB_BLEND_EQUATIONS) ||
      dyn_gfx_state_dirty(cmdbuf, CB_WRITE_MASKS) ||
      dyn_gfx_state_dirty(cmdbuf, CB_BLEND_CONSTANTS) ||
      dyn_gfx_state_dirty(cmdbuf, COLOR_ATTACHMENT_MAP);
   if (blend_dirty) {
      result = build_blend(cmdbuf, fs, &c->blend, &c->blend_count);
      if (result != VK_SUCCESS)
         return result;
   }
   st->blend = c->blend;
   st->blend_count = c->blend_count;

   const bool zsd_dirty =
      !c->valid || fs_user_dirty(cmdbuf) || gfx_state_dirty(cmdbuf, OQ) ||
      dyn_gfx_state_dirty(cmdbuf, DS_DEPTH_TEST_ENABLE) ||
      dyn_gfx_state_dirty(cmdbuf, DS_DEPTH_WRITE_ENABLE) ||
      dyn_gfx_state_dirty(cmdbuf, DS_DEPTH_COMPARE_OP) ||
      dyn_gfx_state_dirty(cmdbuf, DS_STENCIL_TEST_ENABLE) ||
      dyn_gfx_state_dirty(cmdbuf, DS_STENCIL_OP) ||
      dyn_gfx_state_dirty(cmdbuf, DS_STENCIL_COMPARE_MASK) ||
      dyn_gfx_state_dirty(cmdbuf, DS_STENCIL_WRITE_MASK) ||
      dyn_gfx_state_dirty(cmdbuf, DS_STENCIL_REFERENCE) ||
      dyn_gfx_state_dirty(cmdbuf, RS_DEPTH_CLAMP_ENABLE) ||
      dyn_gfx_state_dirty(cmdbuf, RS_DEPTH_CLIP_ENABLE) ||
      dyn_gfx_state_dirty(cmdbuf, RS_DEPTH_BIAS_ENABLE) ||
      dyn_gfx_state_dirty(cmdbuf, RS_DEPTH_BIAS_FACTORS) ||
      dyn_gfx_state_dirty(cmdbuf, INPUT_ATTACHMENT_MAP);
   if (zsd_dirty) {
      result = build_zsd(cmdbuf, &c->zsd);
      if (result != VK_SUCCESS)
         return result;
   }
   st->zsd = c->zsd;

   build_dcd_flags(cmdbuf, fs, draw->prim, &st->flags_0, &st->flags_1);

   /* The depth range comes from the viewport sysvals: set them first. */
   panvk_per_arch(cmd_prepare_draw_sysvals)(cmdbuf, draw, fs);

   panvk_depth_range_all(&cmdbuf->state.gfx, &cmdbuf->vk.dynamic_graphics_state.vp,
                         &st->min_z, &st->max_z);

   struct pan_ptr push_uniforms;
   if (!c->valid || gfx_state_dirty(cmdbuf, VS_PUSH_UNIFORMS)) {
      result = panvk_per_arch(cmd_prepare_gfx_push_uniforms)(cmdbuf, vs,
                                                             &push_uniforms, 1);
      if (result != VK_SUCCESS)
         return result;
      c->vs_fau = push_uniforms.gpu;
   }
   cmdbuf->state.gfx.vs.push_uniforms = c->vs_fau;
   st->vs = vs;
   st->vs_res = cmdbuf->state.gfx.vs.desc.res_table;
   st->vs_fau = c->vs_fau;

   cmdbuf->state.gfx.fs.push_uniforms = 0;
   if (fs) {
      if (!c->valid || !c->fs_fau || fs_user_dirty(cmdbuf) ||
          gfx_state_dirty(cmdbuf, FS_PUSH_UNIFORMS)) {
         result = panvk_per_arch(cmd_prepare_gfx_push_uniforms)(cmdbuf, fs,
                                                                &push_uniforms, 1);
         if (result != VK_SUCCESS)
            return result;
         c->fs_fau = push_uniforms.gpu;
      }
      cmdbuf->state.gfx.fs.push_uniforms = c->fs_fau;
   } else {
      c->fs_fau = 0;
   }

   c->fs = fs;
   c->valid = true;
   return VK_SUCCESS;
}

static VkResult
emit_draw_job(struct panvk_cmd_buffer *cmdbuf, const struct panvk_draw_info *draw,
              const struct v9_draw_state *st, uint32_t layer, struct pan_ptr *out)
{
   struct panvk_batch *batch = cmdbuf->cur_batch;
   const struct panvk_shader_variant *vs = st->vs;
   const struct panvk_shader_variant *fs =
      panvk_shader_only_variant(get_fs(cmdbuf));
   const struct panvk_shader_desc_state *fs_desc = &cmdbuf->state.gfx.fs.desc;
   const struct vk_rasterization_state *rs =
      &cmdbuf->vk.dynamic_graphics_state.rs;
   VkResult result;

   result = panvk_per_arch(cmd_prepare_tiler_context)(cmdbuf, layer);
   if (result != VK_SUCCESS)
      return result;

   struct pan_ptr job = panvk_cmd_alloc_desc(cmdbuf, MALLOC_VERTEX_JOB);
   if (!job.gpu)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;

   const bool points = u_reduced_prim(draw->prim) == MESA_PRIM_POINTS;
   const bool secondary = vs->info.vs.secondary_enable && fs != NULL;
   const bool writes_psiz = vs->info.vs.writes_point_size && points;
   const uint64_t tls = batch->tls.gpu;

   pan_section_pack(job.cpu, MALLOC_VERTEX_JOB, PRIMITIVE, cfg) {
      cfg.draw_mode = translate_prim(draw->prim);
      cfg.index_type = index_size_to_index_type(draw->index.index_size);
      cfg.point_size_array_format = writes_psiz
         ? MALI_POINT_SIZE_ARRAY_FORMAT_FP16
         : MALI_POINT_SIZE_ARRAY_FORMAT_NONE;
      cfg.primitive_index_enable = fs ? fs->info.fs.reads_primitive_id : false;
      cfg.primitive_index_override =
         (vs->info.outputs_written & VARYING_BIT_PRIMITIVE_ID) &&
         cfg.primitive_index_enable;
      cfg.layer_index_enable = vs->info.outputs_written & VARYING_BIT_LAYER;
      cfg.low_depth_cull = cfg.high_depth_cull =
         vk_rasterization_state_depth_clip_enable(rs);
      cfg.secondary_shader = secondary;
      cfg.primitive_restart = draw->index.index_size && draw->index.restart_enable;
      /* Base vertex offset applies to indexed and non-indexed draws alike. */
      cfg.base_vertex_offset = draw->vertex.base;
      cfg.index_count = draw->vertex.count;
   }

   pan_section_pack(job.cpu, MALLOC_VERTEX_JOB, INSTANCE_COUNT, cfg) {
      cfg.count = draw->instance.count;
   }

   pan_section_pack(job.cpu, MALLOC_VERTEX_JOB, ALLOCATION, cfg) {
      if (secondary) {
         unsigned sz = vs->info.varyings.formats.generic_size_B;
         cfg.vertex_packet_stride = sz + 16;
         cfg.vertex_attribute_stride = sz;
      } else {
         /* What the hardware wants for no varyings. */
         cfg.vertex_packet_stride = 16;
         cfg.vertex_attribute_stride = 0;
      }
   }

   pan_section_pack(job.cpu, MALLOC_VERTEX_JOB, TILER, cfg) {
      cfg.address = batch->tiler.ctx.valhall.desc;
   }

   emit_scissor(cmdbuf, pan_section_ptr(job.cpu, MALLOC_VERTEX_JOB, SCISSOR));
   emit_primitive_size(cmdbuf, vs, draw->prim,
                       pan_section_ptr(job.cpu, MALLOC_VERTEX_JOB, PRIMITIVE_SIZE));

   pan_section_pack(job.cpu, MALLOC_VERTEX_JOB, INDICES, cfg) {
      cfg.address = draw->index.index_size
                       ? draw->index.buffer_dev_addr +
                            (uint64_t)draw->index.offset * draw->index.index_size
                       : 0;
   }

   pan_section_pack(job.cpu, MALLOC_VERTEX_JOB, DRAW, cfg) {
      pan_unpack(&st->flags_0, DCD_FLAGS_0, flags_0);
      pan_unpack(&st->flags_1, DCD_FLAGS_1, flags_1);
      cfg.flags_0 = flags_0;
      cfg.flags_1 = flags_1;
      /* Malloc-vertex mode: the hardware fills in where the varyings went. */
      cfg.vertex_array.packet = true;
      cfg.minimum_z = st->min_z;
      cfg.maximum_z = st->max_z;
      cfg.depth_stencil = st->zsd;
      cfg.blend = st->blend;
      cfg.blend_count = st->blend_count;
      cfg.occlusion = cmdbuf->state.gfx.occlusion_query.ptr;

      if (fs) {
         pack_shader_env(&cfg.shader, fs_desc->res_table,
                         panvk_priv_mem_dev_addr(fs->spd), tls,
                         cmdbuf->state.gfx.fs.push_uniforms,
                         fs->fau.total_count);
      }
   }

   const uint64_t pos_spd = points ? panvk_priv_mem_dev_addr(vs->spds.pos_points)
                                   : panvk_priv_mem_dev_addr(vs->spds.pos_triangles);

   pan_section_pack(job.cpu, MALLOC_VERTEX_JOB, POSITION, cfg) {
      pack_shader_env(&cfg, st->vs_res, pos_spd, tls, st->vs_fau, vs->fau.total_count);
   }

   pan_section_pack(job.cpu, MALLOC_VERTEX_JOB, VARYING, cfg) {
      if (secondary) {
         pack_shader_env(&cfg, st->vs_res, panvk_priv_mem_dev_addr(vs->spds.var), tls,
                         st->vs_fau, vs->fau.total_count);
      }
   }

   util_dynarray_append(&batch->jobs, job.cpu);
   *out = job;
   return VK_SUCCESS;
}

/* An indirect draw: the job built for instance 0 and placeholder counts, and in front of it a
 * kernel that reads the command and patches the job, the sysvals and the per-instance attribute
 * offsets. The job waits on the kernel and is not prefetched before the kernel has run. */
static VkResult
patch_indirect_job(struct panvk_cmd_buffer *cmdbuf, const struct panvk_draw_info *draw,
                   uint64_t cmd, struct pan_ptr job)
{
   struct panvk_batch *batch = cmdbuf->cur_batch;
   const struct panvk_shader_variant *vs =
      panvk_shader_hw_variant(cmdbuf->state.gfx.vs.shader);
   const struct vk_dynamic_graphics_state *dyns = &cmdbuf->vk.dynamic_graphics_state;
   const struct vk_vertex_input_state *vi = dyns->vi;

   struct pan_ptr ptr =
      panvk_cmd_alloc_dev_mem(cmdbuf, desc, sizeof(struct libpan_v9_indirect_draw), 8);
   if (!ptr.gpu)
      return VK_ERROR_OUT_OF_DEVICE_MEMORY;

   struct libpan_v9_indirect_draw *d = ptr.cpu;
   *d = (struct libpan_v9_indirect_draw){
      .cmd = cmd,
      .job = job.gpu,
      .index_buffer = draw->index.index_size ? draw->index.buffer_dev_addr : 0,
      .attribs = cmdbuf->state.gfx.vs.desc.driver_set.dev_addr,
      .index_size = draw->index.index_size,
   };

   if (shader_uses_sysval(vs, graphics, vs.first_vertex))
      d->first_vertex_sysval = cmdbuf->state.gfx.vs.push_uniforms +
         shader_remapped_sysval_offset(vs, sysval_offset(graphics, vs.first_vertex));
   if (shader_uses_sysval(vs, graphics, vs.base_instance))
      d->base_instance_sysval = cmdbuf->state.gfx.vs.push_uniforms +
         shader_remapped_sysval_offset(vs, sysval_offset(graphics, vs.base_instance));

   u_foreach_bit(i, vi->attributes_valid) {
      const struct vk_vertex_attribute_state *a = &vi->attributes[i];
      const uint32_t stride = dyns->vi_binding_strides[a->binding];
      if (i < 16 && vi->bindings[a->binding].input_rate == VK_VERTEX_INPUT_RATE_INSTANCE &&
          stride) {
         d->patch_attribs |= BITFIELD_BIT(i);
         d->strides[i] = stride;
      }
   }

   struct panvk_precomp_ctx ctx = panvk_per_arch(precomp_cs)(cmdbuf);
   panlib_v9_draw_indirect(&ctx, panlib_1d(1), PANLIB_BARRIER_NONE, ptr.gpu);
   const unsigned patch_job = batch->vtc_jc.job_index;

   pan_jc_add_job(&batch->vtc_jc, MALI_JOB_TYPE_MALLOC_VERTEX, false, true, 0,
                  patch_job, &job, false);
   return VK_SUCCESS;
}


/* --- geometry shaders (see panvk_vX_shader.c: v9_build_gs_rast) ------------------------------ */

static enum panvk_gs_topology
gs_topology(enum mesa_prim prim)
{
   switch (prim) {
   case MESA_PRIM_LINE_STRIP:
   case MESA_PRIM_TRIANGLE_STRIP:
   case MESA_PRIM_LINE_STRIP_ADJACENCY:
      return PANVK_GS_TOPO_STRIP;
   case MESA_PRIM_TRIANGLE_FAN:
      return PANVK_GS_TOPO_FAN;
   case MESA_PRIM_TRIANGLE_STRIP_ADJACENCY:
      return PANVK_GS_TOPO_TRI_STRIP_ADJ;
   default:
      return PANVK_GS_TOPO_LIST;
   }
}

static uint32_t
gs_prims_per_instance(const struct panvk_draw_info *draw, const struct panvk_shader_variant *gs)
{
   const uint32_t n = draw->vertex.count;
   const uint32_t vpp = MAX2(gs->gs.input_verts_per_prim, 1);

   switch (gs_topology(draw->prim)) {
   case PANVK_GS_TOPO_STRIP:
      return n >= vpp ? n - (vpp - 1) : 0;
   case PANVK_GS_TOPO_FAN:
      return n >= 3 ? n - 2 : 0;
   case PANVK_GS_TOPO_TRI_STRIP_ADJ:
      return n >= 6 ? (n - 4) / 2 : 0;
   default:
      return n / vpp;
   }
}

/* How many vertex records an indexed draw needs: one per index value up to the largest. Read on
 * the CPU when the indices can be; otherwise every vertex the per-vertex buffers hold. */
static uint32_t
gs_index_records(struct panvk_cmd_buffer *cmdbuf, const struct panvk_draw_info *draw)
{
   const uint32_t isz = draw->index.index_size;
   const uint32_t restart = isz == 1 ? 0xff : isz == 2 ? 0xffff : 0xffffffff;

   if (cmdbuf->state.gfx.ib.cpu_readable && !draw->indirect.buffer_dev_addr) {
      const uint8_t *ib = (const uint8_t *)(uintptr_t)draw->index.buffer_dev_addr +
                          (uint64_t)draw->index.offset * isz;
      uint32_t max = 0;
      for (uint32_t i = 0; i < draw->vertex.count; i++) {
         const uint32_t v = isz == 1 ? ib[i] : isz == 2 ? ((const uint16_t *)ib)[i]
                                                         : ((const uint32_t *)ib)[i];
         if (!(draw->index.restart_enable && v == restart))
            max = MAX2(max, v);
      }
      return max + 1;
   }

   const struct vk_dynamic_graphics_state *dyns = &cmdbuf->vk.dynamic_graphics_state;
   const struct vk_vertex_input_state *vi = dyns->vi;
   uint32_t bound = isz == 1 ? 256 : 65536;
   u_foreach_bit(i, vi->attributes_valid) {
      const struct vk_vertex_attribute_state *a = &vi->attributes[i];
      const uint32_t stride = dyns->vi_binding_strides[a->binding];
      if (vi->bindings[a->binding].input_rate != VK_VERTEX_INPUT_RATE_VERTEX || !stride)
         continue;
      const struct panvk_attrib_buf *vb = &cmdbuf->state.gfx.vb.bufs[a->binding];
      bound = MIN2(bound, (uint32_t)(vb->size / stride) + 1);
   }
   return bound;
}

static uint64_t
gs_upload_table(struct panvk_cmd_buffer *cmdbuf, const uint32_t *offsets)
{
   struct pan_ptr t = panvk_cmd_alloc_dev_mem(cmdbuf, desc, 64 * sizeof(uint32_t), 4);
   if (t.cpu)
      memcpy(t.cpu, offsets, 64 * sizeof(uint32_t));
   return t.gpu;
}

/* Transform feedback from the geometry shader emulation: the scan and copy kernels after the
 * geometry shader job, chained through the write offsets like consecutive draws (see
 * jm/panvk_vX_cmd_draw.c, emit_xfb_jobs).
 *
 * group: 0, or the tessellator's vertices per primitive when gs is a tess eval shader, which
 * emits one point per tessellator index: every `group` consecutive invocations are captured as
 * one primitive (see csf/panvk_vX_cmd_draw.c). */
static void
emit_xfb_jobs(struct panvk_cmd_buffer *cmdbuf, const struct panvk_shader_variant *gs,
              uint64_t out_index, unsigned dep_job_id, uint32_t group)
{
   struct panvk_batch *batch = cmdbuf->cur_batch;
   const uint32_t n = cmdbuf->state.gfx.gs.xfb_invocations;
   const uint32_t k_dw = gs->gs.xfb_dwords;

   u_foreach_bit(s, gs->gs.stream_mask) {
      bool captures = false;
      for (unsigned b = 0; b < 4; b++)
         captures |= gs->gs.xfb_strides[b] && gs->gs.xfb_buffer_stream[b] == s;
      if (!captures)
         continue;

      struct pan_ptr mem = panvk_cmd_alloc_dev_mem(
         cmdbuf, desc, sizeof(struct libpan_xfb_draw) + 16 + (k_dw + n) * 4, 16);
      if (!mem.gpu)
         return;

      const uint64_t slot = mem.gpu + sizeof(struct libpan_xfb_draw);
      const uint64_t table = slot + 16;
      struct libpan_xfb_draw *d = mem.cpu;
      *d = (struct libpan_xfb_draw){
         .offsets_in = cmdbuf->state.gfx.xfb.offsets,
         .offsets_out = slot,
         .counts = cmdbuf->state.gfx.gs.xfb_counts + (uint64_t)s * n * 4,
         .first = table + k_dw * 4,
         .indices = s ? cmdbuf->state.gfx.gs.xfb_stream_index +
                           (uint64_t)(s - 1) * cmdbuf->state.gfx.gs.xfb_stream_idx * 4
                      : out_index,
         .staging = cmdbuf->state.gfx.gs.xfb_staging,
         .table = table,
         .invocations = group ? n / group : n,
         .idx_per_invocation =
            group ? group : gs->gs.max_output_prims * gs->gs.output_verts_per_prim,
         .vpp = group ? group : gs->gs.output_verts_per_prim,
         .record_dwords = k_dw,
         .count_stride = group,
         .stats = cmdbuf->state.gfx.xfb.query_stats[s],
      };
      for (unsigned b = 0; b < 4; b++) {
         d->buffer[b] = cmdbuf->state.gfx.xfb.addr[b];
         d->size[b] = cmdbuf->state.gfx.xfb.size[b];
         d->stride[b] = gs->gs.xfb_buffer_stream[b] == s ? gs->gs.xfb_strides[b] : 0;
      }
      memcpy((uint8_t *)mem.cpu + sizeof(struct libpan_xfb_draw) + 16, gs->gs.xfb_table, k_dw * 4);
      cmdbuf->state.gfx.xfb.offsets = slot;

      struct panvk_precomp_ctx precomp_ctx = panvk_per_arch(precomp_cs)(cmdbuf);
      const struct panlib_xfb_scan_args scan = {.d = mem.gpu};
      panlib_xfb_scan_struct(&precomp_ctx,
                             panlib_1d_with_jm_deps(1, dep_job_id, cmdbuf->state.gfx.xfb.last_job),
                             PANLIB_BARRIER_JM_BARRIER, scan);
      const unsigned scanned = batch->vtc_jc.job_index;

      const struct panlib_xfb_copy_args copy = {.d = mem.gpu};
      panlib_xfb_copy_struct(&precomp_ctx, panlib_1d_with_jm_deps(DIV_ROUND_UP(n, 64), scanned, 0),
                             PANLIB_BARRIER_JM_BARRIER, copy);
      cmdbuf->state.gfx.xfb.last_job = batch->vtc_jc.job_index;
   }
}

/* A draw through a geometry shader: the vertex shader as a COMPUTE job writing records, the
 * geometry shader as a COMPUTE job writing output vertices and an index list, and the indexed
 * MALLOC_VERTEX job that rasterizes them through the geometry shader's rasterization variant. */
/* VK_POLYGON_MODE_LINE/POINT on a triangle draw with no geometry shader bound: the draw goes
 * through the emulation with the internal polygon-mode shader (panvk_per_arch(polygon_mode_gs)),
 * between the vertex shader's own PANVK_VS_VARIANT_POLY_* variants. */
static bool
polygon_mode_draw(const struct panvk_cmd_buffer *cmdbuf, const struct panvk_draw_info *draw)
{
   const struct vk_rasterization_state *rs = &cmdbuf->vk.dynamic_graphics_state.rs;
   const struct panvk_shader *vs = cmdbuf->state.gfx.vs.shader;

   return rs->polygon_mode != VK_POLYGON_MODE_FILL && !rs->rasterizer_discard_enable &&
          (draw->prim == MESA_PRIM_TRIANGLES || draw->prim == MESA_PRIM_TRIANGLE_STRIP ||
           draw->prim == MESA_PRIM_TRIANGLE_FAN) &&
          vs && vs->variants[PANVK_VS_VARIANT_POLY_REC].bin_size &&
          vs->variants[PANVK_VS_VARIANT_POLY_RAST].bin_size;
}

/* Transform feedback from a vertex shader with no geometry shader bound: the draw goes through
 * the emulation with the internal pass-through shader (panvk_per_arch(xfb_passthrough_gs)), as
 * a polygon-mode draw does. */
static bool
vs_xfb_draw(const struct panvk_cmd_buffer *cmdbuf)
{
   const struct panvk_shader *vs = cmdbuf->state.gfx.vs.shader;

   return cmdbuf->state.gfx.xfb.active && vs &&
          vs->variants[PANVK_VS_VARIANT_POLY_REC].gs.vs_xfb_size &&
          vs->variants[PANVK_VS_VARIANT_POLY_REC].bin_size &&
          vs->variants[PANVK_VS_VARIANT_POLY_RAST].bin_size;
}

enum gs_emulation {
   GS_EMULATION_NONE,
   GS_EMULATION_APP,
   GS_EMULATION_POLYGON_MODE,
   GS_EMULATION_XFB,
};

/* After a kernel that rewrote a job descriptor: the job manager caches descriptors, so it is
 * cleaned and invalidated along with L2 before the patched job is read (as on Bifrost,
 * jm/panvk_vX_cmd_draw.c). The tiler is cleaned, never invalidated: it holds the polygon lists
 * of the batch's earlier draws. Returns the flush job's index, 0 on failure. */
static unsigned
emit_cache_flush_job(struct panvk_cmd_buffer *cmdbuf, unsigned dep_job_id)
{
   struct panvk_batch *batch = cmdbuf->cur_batch;
   struct pan_ptr job = panvk_cmd_alloc_desc(cmdbuf, CACHE_FLUSH_JOB);
   if (!job.gpu)
      return 0;

   pan_section_pack(job.cpu, CACHE_FLUSH_JOB, PAYLOAD, cfg) {
      cfg.clean_shader_core_ls = true;
      cfg.invalidate_shader_core_ls = true;
      cfg.invalidate_shader_core_other = true;
      cfg.job_manager_clean = true;
      cfg.job_manager_invalidate = true;
      cfg.tiler_clean = true;
      cfg.tiler_invalidate = false;
      cfg.l2_clean = true;
      cfg.l2_invalidate = true;
   }

   util_dynarray_append(&batch->jobs, job.cpu);
   return pan_jc_add_job(&batch->vtc_jc, MALI_JOB_TYPE_CACHE_FLUSH, true, true, dep_job_id, 0,
                         &job, false);
}

/* An indirect draw through the emulation: the kernel that sizes `job` (a COMPUTE job added
 * after it with suppress_prefetch) from what the setup kernel `dep` wrote at dims, and, with
 * cmd, offsets the per-instance attributes by firstInstance, then a cache flush so the job
 * manager reads the patched descriptor. Returns the flush's job index, 0 on failure. */
static unsigned
emit_gs_job_patch(struct panvk_cmd_buffer *cmdbuf, uint64_t dims, uint64_t job, uint64_t cmd,
                  bool indexed, unsigned dep)
{
   struct panvk_batch *batch = cmdbuf->cur_batch;
   const struct vk_dynamic_graphics_state *dyns = &cmdbuf->vk.dynamic_graphics_state;
   const struct vk_vertex_input_state *vi = dyns->vi;

   struct pan_ptr ptr =
      panvk_cmd_alloc_dev_mem(cmdbuf, desc, sizeof(struct libpan_v9_gs_job), 8);
   if (!ptr.gpu)
      return 0;

   struct libpan_v9_gs_job *j = ptr.cpu;
   *j = (struct libpan_v9_gs_job){
      .dims = dims,
      .job = job,
      .cmd = cmd,
      .attribs = cmdbuf->state.gfx.vs.desc.driver_set.dev_addr,
      .indexed = indexed,
   };
   if (cmd) {
      u_foreach_bit(i, vi->attributes_valid) {
         const struct vk_vertex_attribute_state *a = &vi->attributes[i];
         const uint32_t stride = dyns->vi_binding_strides[a->binding];
         if (i < 16 && vi->bindings[a->binding].input_rate == VK_VERTEX_INPUT_RATE_INSTANCE &&
             stride) {
            j->patch_attribs |= BITFIELD_BIT(i);
            j->strides[i] = stride;
         }
      }
   }

   struct panvk_precomp_ctx ctx = panvk_per_arch(precomp_cs)(cmdbuf);
   /* suppress_prefetch: the job after it is only valid once this has run. */
   panlib_v9_gs_job(&ctx, panlib_1d_with_jm_deps(1, dep, 0),
                    PANLIB_BARRIER_JM_BARRIER | PANLIB_BARRIER_JM_SUPPRESS_PREFETCH, ptr.gpu);
   return emit_cache_flush_job(cmdbuf, batch->vtc_jc.job_index);
}

/* panlib_gs_indirect_setup for one draw of the indirect buffer. It waits for everything before
 * it, so a command written earlier in the batch is in place. Returns its job index. */
static unsigned
emit_gs_indirect_setup(struct panvk_cmd_buffer *cmdbuf, uint64_t setup)
{
   struct panvk_precomp_ctx ctx = panvk_per_arch(precomp_cs)(cmdbuf);
   const struct panlib_gs_indirect_setup_args args = {.d = setup};
   panlib_gs_indirect_setup_struct(&ctx, panlib_1d(1), PANLIB_BARRIER_JM_BARRIER, args);
   return cmdbuf->cur_batch->vtc_jc.job_index;
}

/* An indirect draw (draw->indirect.buffer_dev_addr, one command): its size is only known on the
 * GPU, so the emulation is sized for PANVK_GS_INDIRECT_MAX_PRIMS input primitives, and
 * panlib_gs_indirect_setup cuts the draw to that and sizes the vertex job (as on CSF). */
static void
gs_draw(struct panvk_cmd_buffer *cmdbuf, const struct panvk_draw_info *draw,
        enum gs_emulation mode)
{
   struct panvk_device *dev = to_panvk_device(cmdbuf->vk.base.device);
   const struct vk_rasterization_state *rs = &cmdbuf->vk.dynamic_graphics_state.rs;
   /* The internal shaders run between the vertex shader's own PANVK_VS_VARIANT_POLY_* variants. */
   const bool poly = mode != GS_EMULATION_APP;
   const struct panvk_shader *gs_shader =
      mode == GS_EMULATION_POLYGON_MODE
         ? panvk_per_arch(polygon_mode_gs)(dev, rs->polygon_mode == VK_POLYGON_MODE_POINT)
      : mode == GS_EMULATION_XFB
         ? panvk_per_arch(xfb_passthrough_gs)(
              dev, &cmdbuf->state.gfx.vs.shader->variants[PANVK_VS_VARIANT_POLY_REC], draw->prim)
         : cmdbuf->state.gfx.gs.app_shader;
   if (!gs_shader)
      return;
   const struct panvk_shader_variant *gs = panvk_shader_only_variant(gs_shader);
   const struct panvk_shader_variant *vs =
      poly ? &cmdbuf->state.gfx.vs.shader->variants[PANVK_VS_VARIANT_POLY_REC]
           : panvk_shader_hw_variant(cmdbuf->state.gfx.vs.shader);
   const struct panvk_shader_variant *rast =
      poly ? &cmdbuf->state.gfx.vs.shader->variants[PANVK_VS_VARIANT_POLY_RAST]
           : panvk_shader_gs_rast_variant(gs_shader);
   /* The layout of the emulation's output vertices: the geometry shader's own, or, for the
    * internal shaders, which copy their input through, the vertex shader's. */
   const struct panvk_shader_variant *out_rec = poly ? vs : gs;

   if (!vs->gs.vs_for_gs || !panvk_priv_mem_check_alloc(gs->spd) ||
       !panvk_priv_mem_check_alloc(rast->spd))
      return;

   const bool indexed = draw->index.index_size != 0;
   const bool indirect = draw->indirect.buffer_dev_addr != 0;
   const uint32_t vpp = MAX2(gs->gs.input_verts_per_prim, 1);
   const uint32_t max_vertices = PANVK_GS_INDIRECT_MAX_PRIMS * vpp + vpp;
   uint32_t inst = draw->instance.count;
   uint32_t recs = indexed ? gs_index_records(cmdbuf, draw) : draw->vertex.count;
   uint32_t prims_per_instance = gs_prims_per_instance(draw, gs);
   uint32_t in_prims = prims_per_instance * inst;
   if (indirect) {
      /* Records at the padded stride of the largest vertex job, for as many instances as twice
       * the primitive cap could need. */
      recs = pan_padded_vertex_count(indexed ? recs : max_vertices);
      inst = MAX2(1, 2 * PANVK_GS_INDIRECT_MAX_PRIMS * vpp / recs);
      prims_per_instance = 0;
      in_prims = PANVK_GS_INDIRECT_MAX_PRIMS;
   }
   if (!in_prims || !recs)
      return;

   const uint32_t invocations = in_prims * gs->gs.invocations;
   const uint32_t out_verts = invocations * gs->gs.max_output_verts;
   const uint32_t out_indices =
      invocations * gs->gs.max_output_prims * gs->gs.output_verts_per_prim;
   if (!out_verts || !out_indices)
      return;

   struct pan_ptr in_pos = panvk_cmd_alloc_dev_mem(cmdbuf, desc, recs * inst * 16, 64);
   struct pan_ptr in_gen =
      vs->gs.rec_stride ? panvk_cmd_alloc_dev_mem(cmdbuf, desc, recs * inst * vs->gs.rec_stride, 64)
                        : (struct pan_ptr){0};
   struct pan_ptr out_pos = panvk_cmd_alloc_dev_mem(cmdbuf, desc, out_verts * 16, 64);
   struct pan_ptr out_gen =
      out_rec->gs.rec_stride
         ? panvk_cmd_alloc_dev_mem(cmdbuf, desc, out_verts * out_rec->gs.rec_stride, 64)
         : (struct pan_ptr){0};
   struct pan_ptr out_idx = panvk_cmd_alloc_dev_mem(cmdbuf, desc, out_indices * 4, 4);
   struct pan_ptr setup =
      indirect ? panvk_cmd_alloc_dev_mem(cmdbuf, desc, sizeof(struct libpan_gs_indirect), 64)
               : (struct pan_ptr){0};
   if (indirect && !setup.gpu)
      return;
   /* Transform feedback: a primitive count per emulation invocation and a capture record per
    * output vertex slot, and (geometryStreams) a copy of both plus an index list for every
    * stream past 0 up to the last one emitted on. */
   cmdbuf->state.gfx.gs.xfb_counts = 0;
   cmdbuf->state.gfx.gs.xfb_staging = 0;
   cmdbuf->state.gfx.gs.xfb_invocations = 0;
   cmdbuf->state.gfx.gs.xfb_stream_index = 0;
   cmdbuf->state.gfx.gs.xfb_stream_slots = 0;
   cmdbuf->state.gfx.gs.xfb_stream_idx = 0;
   if (cmdbuf->state.gfx.xfb.active && gs->gs.xfb_dwords) {
      const unsigned nstreams = MAX2(util_last_bit(gs->gs.stream_mask), 1);
      struct pan_ptr counts =
         panvk_cmd_alloc_dev_mem(cmdbuf, desc, nstreams * invocations * 4, 4);
      struct pan_ptr staging = panvk_cmd_alloc_dev_mem(
         cmdbuf, desc, nstreams * out_verts * gs->gs.xfb_dwords * 4, 4);
      struct pan_ptr stream_index =
         nstreams > 1
            ? panvk_cmd_alloc_dev_mem(cmdbuf, desc, (nstreams - 1) * out_indices * 4, 4)
            : (struct pan_ptr){0};
      if (!counts.gpu || !staging.gpu || (nstreams > 1 && !stream_index.gpu))
         return;
      cmdbuf->state.gfx.gs.xfb_counts = counts.gpu;
      cmdbuf->state.gfx.gs.xfb_staging = staging.gpu;
      cmdbuf->state.gfx.gs.xfb_invocations = invocations;
      cmdbuf->state.gfx.gs.xfb_stream_index = stream_index.gpu;
      cmdbuf->state.gfx.gs.xfb_stream_slots = out_verts;
      cmdbuf->state.gfx.gs.xfb_stream_idx = out_indices;
   }

   const uint64_t in_offsets = gs_upload_table(cmdbuf, vs->gs.rec_offsets);
   const uint64_t out_offsets = gs_upload_table(cmdbuf, out_rec->gs.rec_offsets);
   if (!in_pos.gpu || !out_pos.gpu || !out_idx.gpu || !in_offsets || !out_offsets ||
       (vs->gs.rec_stride && !in_gen.gpu) || (out_rec->gs.rec_stride && !out_gen.gpu))
      return;

   /* The vertex shader: descriptors, sysvals and push uniforms as for any draw, then its
    * records. */
   cmdbuf->state.gfx.sysvals.gs_io.pos = in_pos.gpu;
   cmdbuf->state.gfx.sysvals.gs_io.general = in_gen.gpu;
   cmdbuf->state.gfx.sysvals.gs_io.instance_stride = recs;

   const struct panvk_draw_info rast_draw = {
      .index.buffer_dev_addr = out_idx.gpu,
      .index.buffer_size = out_indices * 4,
      .index.index_size = 4,
      .vertex.count = out_indices,
      .instance.count = 1,
      .prim = gs->gs.output_verts_per_prim == 3   ? MESA_PRIM_TRIANGLES
              : gs->gs.output_verts_per_prim == 2 ? MESA_PRIM_LINES
                                                  : MESA_PRIM_POINTS,
   };

   struct v9_draw_state st;
   if (prepare_draw(cmdbuf, draw, &st) != VK_SUCCESS)
      return;
   build_dcd_flags(cmdbuf, panvk_shader_only_variant(get_fs(cmdbuf)), rast_draw.prim,
                   &st.flags_0, &st.flags_1);
   /* prepare_draw laid the push uniforms out for the HW variant. */
   if (poly) {
      struct pan_ptr vs_pu;
      if (panvk_per_arch(cmd_prepare_gfx_push_uniforms)(cmdbuf, vs, &vs_pu, 1) != VK_SUCCESS)
         return;
      st.vs_fau = vs_pu.gpu;
   }

   struct panvk_batch *batch = cmdbuf->cur_batch;

   if (indirect) {
      struct libpan_gs_indirect *d = setup.cpu;
      *d = (struct libpan_gs_indirect){
         .pos_desc = in_pos.gpu,
         .gen_desc = in_gen.gpu,
         .cmd = draw->indirect.buffer_dev_addr,
         .fau = st.vs_fau,
         .indexed = indexed,
         .index_records = indexed ? gs_index_records(cmdbuf, draw) : 0,
         .max_vertices = max_vertices,
         .records = recs * inst,
         .first_vertex_off =
            shader_uses_sysval(vs, graphics, vs.first_vertex)
               ? shader_remapped_sysval_offset(vs, sysval_offset(graphics, vs.first_vertex))
               : ~0u,
         .base_instance_off =
            shader_uses_sysval(vs, graphics, vs.base_instance)
               ? shader_remapped_sysval_offset(vs, sysval_offset(graphics, vs.base_instance))
               : ~0u,
         .stride_off =
            shader_uses_sysval(vs, graphics, gs_io.instance_stride)
               ? shader_remapped_sysval_offset(vs,
                                               sysval_offset(graphics, gs_io.instance_stride))
               : ~0u,
      };
   }

   struct pan_ptr vjob = panvk_cmd_alloc_desc(cmdbuf, COMPUTE_JOB);
   if (!vjob.gpu)
      return;
   pan_section_pack(vjob.cpu, COMPUTE_JOB, PAYLOAD, cfg) {
      cfg.workgroup_size_x = 1;
      cfg.workgroup_size_y = 1;
      cfg.workgroup_size_z = 1;
      cfg.workgroup_count_x = recs;
      cfg.workgroup_count_y = inst;
      cfg.workgroup_count_z = 1;
      cfg.allow_merging_workgroups = true;
      cfg.task_increment = 1;
      cfg.task_axis = MALI_TASK_AXIS_Z;
      cfg.compute.attribute_offset = draw->vertex.base;
      cfg.compute.resources = st.vs_res;
      cfg.compute.shader = panvk_priv_mem_dev_addr(vs->spds.pos_points);
      cfg.compute.thread_storage = batch->tls.gpu;
      cfg.compute.fau = st.vs_fau;
      cfg.compute.fau_count = vs->fau.total_count;
   }
   util_dynarray_append(&batch->jobs, vjob.cpu);
   unsigned patch_id = 0;
   if (indirect) {
      patch_id = emit_gs_job_patch(cmdbuf, setup.gpu, vjob.gpu, draw->indirect.buffer_dev_addr,
                                   indexed, emit_gs_indirect_setup(cmdbuf, setup.gpu));
      if (!patch_id)
         return;
   }
   const unsigned vjob_id = pan_jc_add_job(&batch->vtc_jc, MALI_JOB_TYPE_COMPUTE, false, indirect,
                                           patch_id, 0, &vjob, false);

   /* The geometry shader, as compute with the emulation's push block. */
   struct panvk_descriptor_state *desc_state = &cmdbuf->state.gfx.desc_state;
   struct panvk_shader_desc_state gs_desc = {0};
   const uint32_t ds_count = gs_shader->desc_info.dyn_bufs.count + 1;
   struct pan_ptr ds = panvk_cmd_alloc_dev_mem(cmdbuf, desc, ds_count * PANVK_DESCRIPTOR_SIZE,
                                               PANVK_DESCRIPTOR_SIZE);
   if (!ds.gpu)
      return;
   pan_cast_and_pack(ds.cpu, SAMPLER, cfg) {
      cfg.clamp_integer_array_indices = false;
   }
   panvk_per_arch(cmd_fill_dyn_bufs)(desc_state, &gs_shader->desc_info,
                                     (struct mali_buffer_packed *)((uint8_t *)ds.cpu +
                                                                   PANVK_DESCRIPTOR_SIZE));
   gs_desc.driver_set.dev_addr = ds.gpu;
   gs_desc.driver_set.size = ds_count * PANVK_DESCRIPTOR_SIZE;
   if (panvk_per_arch(cmd_prepare_push_descs)(cmdbuf, desc_state,
                                              gs_shader->desc_info.used_set_mask) != VK_SUCCESS ||
       panvk_per_arch(cmd_prepare_shader_res_table)(cmdbuf, desc_state, &gs_shader->desc_info,
                                                    &gs_desc, 1) != VK_SUCCESS)
      return;

   const uint32_t wg = DIV_ROUND_UP(invocations, PANVK_GS_WORKGROUP_SIZE);
   struct panvk_compute_sysvals sysvals = {0};
   sysvals.num_work_groups.x = wg;
   sysvals.local_group_size.x = PANVK_GS_WORKGROUP_SIZE;
   sysvals.gs = (struct panvk_gs_push){
      .in_pos = in_pos.gpu,
      .in_general = in_gen.gpu,
      .out_pos = out_pos.gpu,
      .out_general = out_gen.gpu,
      .out_index = out_idx.gpu,
      .in_offsets = in_offsets,
      .out_offsets = out_offsets,
      .in_general_stride = vs->gs.rec_stride,
      .out_general_stride = out_rec->gs.rec_stride,
      .num_input_prims = in_prims,
      .index_buf = indexed ? draw->index.buffer_dev_addr +
                                (uint64_t)draw->index.offset * draw->index.index_size
                           : 0,
      .indirect_cmd = indirect ? setup.gpu + offsetof(struct libpan_gs_indirect, out_cmd) : 0,
      .vertex_dcd = indirect ? setup.gpu + offsetof(struct libpan_gs_indirect, dcd) : 0,
      .index_min = indirect ? setup.gpu + offsetof(struct libpan_gs_indirect, zero) : 0,
      .in_pos_desc = indirect ? setup.gpu + offsetof(struct libpan_gs_indirect, pos_desc) : 0,
      .in_general_desc =
         indirect ? setup.gpu + offsetof(struct libpan_gs_indirect, gen_desc) : 0,
      .index_size = draw->index.index_size,
      .topology = gs_topology(draw->prim),
      .prims_per_instance = prims_per_instance,
      .instance_stride = inst > 1 ? recs : 0,
      .out_clip_space = 1,
      .viewport_scale = {cmdbuf->state.gfx.sysvals.viewport.scale.x,
                         cmdbuf->state.gfx.sysvals.viewport.scale.y,
                         cmdbuf->state.gfx.sysvals.viewport.scale.z},
      .viewport_offset = {cmdbuf->state.gfx.sysvals.viewport.offset.x,
                          cmdbuf->state.gfx.sysvals.viewport.offset.y,
                          cmdbuf->state.gfx.sysvals.viewport.offset.z},
      .xfb_counts = cmdbuf->state.gfx.gs.xfb_counts,
      .xfb_staging = cmdbuf->state.gfx.gs.xfb_staging,
      .xfb_invocations = cmdbuf->state.gfx.gs.xfb_invocations,
      .xfb_stream_index = cmdbuf->state.gfx.gs.xfb_stream_index,
      .xfb_stream_slots = cmdbuf->state.gfx.gs.xfb_stream_slots,
      .xfb_stream_idx = cmdbuf->state.gfx.gs.xfb_stream_idx,
      .viewports = cmdbuf->state.gfx.gs.viewports,
      .polygon_cull = ((rs->cull_mode & VK_CULL_MODE_FRONT_BIT) ? PANVK_GS_CULL_FRONT : 0) |
                      ((rs->cull_mode & VK_CULL_MODE_BACK_BIT) ? PANVK_GS_CULL_BACK : 0) |
                      (rs->front_face == VK_FRONT_FACE_COUNTER_CLOCKWISE ? PANVK_GS_FRONT_CCW : 0),
   };

   struct pan_ptr gs_pu;
   if (panvk_per_arch(cmd_prepare_gs_push_uniforms)(cmdbuf, gs, &sysvals, &gs_pu) != VK_SUCCESS)
      return;
   struct pan_compute_dim dim = {wg, 1, 1};
   const uint64_t tls = panvk_per_arch(cmd_dispatch_prepare_tls)(cmdbuf, gs, &dim, false);

   struct pan_ptr gjob = panvk_cmd_alloc_desc(cmdbuf, COMPUTE_JOB);
   if (!gjob.gpu)
      return;
   panvk_per_arch(jm_pack_compute_payload)(gjob.cpu, gs, wg, 1, 1, gs_desc.res_table, gs_pu.gpu,
                                           tls, gs->fau.total_count);
   util_dynarray_append(&batch->jobs, gjob.cpu);
   const unsigned gjob_id = pan_jc_add_job(&batch->vtc_jc, MALI_JOB_TYPE_COMPUTE, true, false,
                                           vjob_id, 0, &gjob, false);
   if (cmdbuf->state.gfx.gs.xfb_counts)
      emit_xfb_jobs(cmdbuf, gs, out_idx.gpu, gjob_id, 0);

   /* The rasterization pass: the emulation's vertices through the rasterization variant. */
   cmdbuf->state.gfx.sysvals.gs_io.pos = out_pos.gpu;
   cmdbuf->state.gfx.sysvals.gs_io.general = out_gen.gpu;
   struct pan_ptr rast_pu;
   if (panvk_per_arch(cmd_prepare_gfx_push_uniforms)(cmdbuf, rast, &rast_pu, 1) != VK_SUCCESS)
      return;
   st.vs = rast;
   st.vs_res = 0;
   st.vs_fau = rast_pu.gpu;

   uint32_t view_mask = cmdbuf->state.gfx.render.view_mask;
   uint32_t layer_count = view_mask ? util_bitcount(view_mask)
                                    : cmdbuf->state.gfx.render.layer_count;
   for (uint32_t i = 0; i < layer_count; i++) {
      const uint32_t layer = view_mask ? u_bit_scan(&view_mask) : i;
      struct pan_ptr job;
      if (emit_draw_job(cmdbuf, &rast_draw, &st, layer, &job) != VK_SUCCESS)
         return;
      pan_jc_add_job(&batch->vtc_jc, MALI_JOB_TYPE_MALLOC_VERTEX, true, false, gjob_id, 0,
                     &job, false);
   }

   cmdbuf->state.v9.valid = false;
   clear_dirty_after_draw(cmdbuf);
}

/* The tess mode: either stage may declare it, the tess eval shader wins. */
static void
tess_mode(const struct panvk_cmd_buffer *cmdbuf, enum tess_primitive_mode *prim,
          enum gl_tess_spacing *spacing, bool *ccw, bool *points)
{
   const struct panvk_shader_variant *tcs =
      panvk_shader_only_variant(cmdbuf->state.gfx.tess.tcs);
   const struct panvk_shader_variant *tes =
      panvk_shader_only_variant(cmdbuf->state.gfx.tess.tes);
   *prim = tes->gs.tess_prim ? tes->gs.tess_prim : tcs->gs.tess_prim;
   *spacing = tes->gs.tess_spacing ? tes->gs.tess_spacing : tcs->gs.tess_spacing;
   *ccw = tes->gs.tess_ccw || tcs->gs.tess_ccw;
   *points = tes->gs.tess_point_mode || tcs->gs.tess_point_mode;
}

/* A shader the emulation runs as compute: the bound graphics sets through a resource table of
 * its own (as in gs_draw), and the emulation's push block. */
static bool
emu_prepare(struct panvk_cmd_buffer *cmdbuf, const struct panvk_shader *shader,
            const struct panvk_shader_variant *v, const struct panvk_compute_sysvals *sysvals,
            uint64_t *res_table, uint64_t *fau)
{
   struct panvk_descriptor_state *desc_state = &cmdbuf->state.gfx.desc_state;
   struct panvk_shader_desc_state sd = {0};
   const uint32_t ds_count = shader->desc_info.dyn_bufs.count + 1;
   struct pan_ptr ds = panvk_cmd_alloc_dev_mem(cmdbuf, desc, ds_count * PANVK_DESCRIPTOR_SIZE,
                                               PANVK_DESCRIPTOR_SIZE);
   if (!ds.gpu)
      return false;
   pan_cast_and_pack(ds.cpu, SAMPLER, cfg) {
      cfg.clamp_integer_array_indices = false;
   }
   panvk_per_arch(cmd_fill_dyn_bufs)(desc_state, &shader->desc_info,
                                     (struct mali_buffer_packed *)((uint8_t *)ds.cpu +
                                                                   PANVK_DESCRIPTOR_SIZE));
   sd.driver_set.dev_addr = ds.gpu;
   sd.driver_set.size = ds_count * PANVK_DESCRIPTOR_SIZE;
   if (panvk_per_arch(cmd_prepare_push_descs)(cmdbuf, desc_state,
                                              shader->desc_info.used_set_mask) != VK_SUCCESS ||
       panvk_per_arch(cmd_prepare_shader_res_table)(cmdbuf, desc_state, &shader->desc_info, &sd,
                                                    1) != VK_SUCCESS)
      return false;

   struct pan_ptr pu;
   if (panvk_per_arch(cmd_prepare_gs_push_uniforms)(cmdbuf, v, sysvals, &pu) != VK_SUCCESS)
      return false;
   *res_table = sd.res_table;
   *fau = pu.gpu;
   return true;
}

/* One stage of the tessellation emulation as a COMPUTE job. Each one waits for every job before
 * it: the stages read what the one before wrote, and the tessellation heap is shared by every
 * tessellated draw of the batch. Returns the job index, 0 on failure. */
static unsigned
emu_job(struct panvk_cmd_buffer *cmdbuf, const struct panvk_shader_variant *v, uint32_t wg_x,
        uint32_t wg_y, bool indirect, uint64_t res_table, uint64_t fau, unsigned dep,
        struct pan_ptr *out)
{
   struct panvk_batch *batch = cmdbuf->cur_batch;
   const uint32_t wls = batch->tlsinfo.wls.size;
   const __typeof__(batch->tlsinfo.wls.instances) wls_instances = batch->tlsinfo.wls.instances;
   const __typeof__(batch->wls_total_size) wls_total = batch->wls_total_size;
   struct pan_compute_dim dim = {wg_x, wg_y, 1};
   const uint64_t tls = panvk_per_arch(cmd_dispatch_prepare_tls)(cmdbuf, v, &dim, indirect);
   /* prepare_tls sets the batch's local storage to this shader's: keep the largest. */
   if (batch->tlsinfo.wls.size < wls) {
      batch->tlsinfo.wls.size = wls;
      batch->tlsinfo.wls.instances = wls_instances;
      batch->wls_total_size = wls_total;
   }

   struct pan_ptr job = panvk_cmd_alloc_desc(cmdbuf, COMPUTE_JOB);
   if (!job.gpu)
      return 0;
   panvk_per_arch(jm_pack_compute_payload)(job.cpu, v, wg_x, wg_y, 1, res_table, fau, tls,
                                           v->fau.total_count);
   util_dynarray_append(&batch->jobs, job.cpu);
   if (out)
      *out = job;
   return pan_jc_add_job(&batch->vtc_jc, MALI_JOB_TYPE_COMPUTE, true, indirect, dep, 0, &job,
                         false);
}

/*
 * A tessellated draw, as on CSF (csf/panvk_vX_cmd_draw.c, tess_draw): the vertex shader writes
 * records as for a geometry shader, then
 *
 *   tess control      compute, one workgroup per patch, one invocation per output vertex
 *   panlib_tess_*     COUNT, prefix_sum, WITH_COUNTS: domain points and indices in dev->tess_heap
 *   tess eval         the emulation, one invocation per tessellator index
 *   geometry          the application's, if bound, over the tessellator's primitives
 *
 * and the rasterization pass draws the last stage's records. The heap is the batch's, as on
 * Bifrost (jm/panvk_vX_cmd_draw.c): the first tessellated draw of a batch rewinds it, and every
 * job here waits for all the jobs before it.
 *
 * An indirect draw is one command of the indirect buffer: its vertex pass is sized by
 * panlib_gs_indirect_setup, the patch counts and the tess control grid by
 * panlib_tess_indirect_setup, for at most PANVK_TESS_INDIRECT_GRID patches.
 */
static void
tess_draw(struct panvk_cmd_buffer *cmdbuf, const struct panvk_draw_info *draw)
{
   struct panvk_device *dev = to_panvk_device(cmdbuf->vk.base.device);
   const struct vk_dynamic_graphics_state *dyn = &cmdbuf->vk.dynamic_graphics_state;
   const struct vk_rasterization_state *rs = &dyn->rs;
   const struct panvk_shader *tcs_shader = cmdbuf->state.gfx.tess.tcs;
   const struct panvk_shader *tes_shader = cmdbuf->state.gfx.tess.tes;
   const struct panvk_shader *gs_shader = cmdbuf->state.gfx.gs.app_shader;

   if (!tcs_shader || !tes_shader || !dev->tess_heap)
      return;

   const struct panvk_shader_variant *vs = panvk_shader_hw_variant(cmdbuf->state.gfx.vs.shader);
   const struct panvk_shader_variant *tcs = panvk_shader_only_variant(tcs_shader);
   const struct panvk_shader_variant *tes = panvk_shader_only_variant(tes_shader);
   const struct panvk_shader_variant *gs =
      gs_shader ? panvk_shader_only_variant(gs_shader) : NULL;
   const struct panvk_shader_variant *rast =
      panvk_shader_gs_rast_variant(gs_shader ? gs_shader : tes_shader);

   if (!vs->gs.vs_for_gs || !panvk_priv_mem_check_alloc(tcs->spd) ||
       !panvk_priv_mem_check_alloc(tes->spd) || (gs && !panvk_priv_mem_check_alloc(gs->spd)) ||
       !panvk_priv_mem_check_alloc(rast->spd))
      return;

   const bool indexed = draw->index.index_size != 0;
   const bool indirect = draw->indirect.buffer_dev_addr != 0;
   const uint32_t pcp = MAX2(dyn->ts.patch_control_points, 1);
   const uint32_t max_vertices = PANVK_TESS_INDIRECT_GRID * pcp;
   /* Indirect: the counts below are allocation sizes; the GPU writes the real ones. */
   const uint32_t ppi = indirect ? 1 : draw->vertex.count / pcp;
   uint32_t inst = draw->instance.count;
   uint32_t recs = indexed ? gs_index_records(cmdbuf, draw) : draw->vertex.count;
   if (indirect) {
      recs = pan_padded_vertex_count(indexed ? recs : max_vertices);
      inst = MAX2(1, 2 * max_vertices / recs);
   }
   const uint32_t nr_patches = indirect ? PANVK_TESS_INDIRECT_GRID : ppi * inst;
   const uint32_t tcs_stride_el = tcs->gs.tcs_output_stride / 4;
   if (!nr_patches || !recs || (uint64_t)nr_patches * tcs->gs.tcs_output_stride > (64u << 20))
      return;

   /* 64 is maxTessellationGenerationLevel, where a quad domain is 64x64 quads of two
    * triangles. */
   const uint32_t cap =
      indirect ? PANVK_TESS_MAX_INDICES
               : MIN2((uint64_t)nr_patches * 64 * 64 * 6, PANVK_TESS_MAX_INDICES);

   enum tess_primitive_mode prim;
   enum gl_tess_spacing spacing;
   bool ccw, points;
   tess_mode(cmdbuf, &prim, &spacing, &ccw, &points);
   const uint32_t tess_vpp = points ? 1 : prim == TESS_PRIMITIVE_ISOLINES ? 2 : 3;

   /* The geometry shader, if any, runs once per (tessellator primitive, invocation). */
   const uint32_t gs_prims = gs ? cap / MAX2(gs->gs.input_verts_per_prim, 1) : 0;
   const uint32_t gs_invocations = gs ? gs_prims * gs->gs.invocations : 0;
   const uint32_t out_verts = gs ? gs_invocations * gs->gs.max_output_verts : cap;
   const uint32_t out_indices =
      gs ? gs_invocations * gs->gs.max_output_prims * gs->gs.output_verts_per_prim : cap;
   if (gs && (gs->gs.input_verts_per_prim != tess_vpp || !out_verts || !out_indices))
      return;

   /* [tcs records][coord_allocs][counts][out_draws: 32][out: 16][params] */
   uint32_t size = nr_patches * tcs_stride_el * 4;
   const uint32_t coord_off = ALIGN_POT(size, 16);
   size = coord_off + nr_patches * 4;
   const uint32_t counts_off = ALIGN_POT(size, 16);
   size = counts_off + nr_patches * 4;
   const uint32_t draws_off = ALIGN_POT(size, 16);
   const uint32_t out_off = draws_off + 32;
   const uint32_t params_off = ALIGN_POT(out_off + 16, 16);
   size = params_off + sizeof(struct poly_tess_params);

   struct pan_ptr blob = panvk_cmd_alloc_dev_mem(cmdbuf, desc, size, 64);
   struct pan_ptr in_pos = panvk_cmd_alloc_dev_mem(cmdbuf, desc, recs * inst * 16, 64);
   struct pan_ptr in_gen =
      vs->gs.rec_stride ? panvk_cmd_alloc_dev_mem(cmdbuf, desc, recs * inst * vs->gs.rec_stride, 64)
                        : (struct pan_ptr){0};
   struct pan_ptr tes_pos = panvk_cmd_alloc_dev_mem(cmdbuf, desc, cap * 16, 64);
   struct pan_ptr tes_gen =
      tes->gs.rec_stride ? panvk_cmd_alloc_dev_mem(cmdbuf, desc, cap * tes->gs.rec_stride, 64)
                         : (struct pan_ptr){0};
   struct pan_ptr tes_idx = panvk_cmd_alloc_dev_mem(cmdbuf, desc, cap * 4, 4);
   struct pan_ptr gs_pos = {0}, gs_gen = {0}, gs_idx = {0};
   if (gs) {
      gs_pos = panvk_cmd_alloc_dev_mem(cmdbuf, desc, out_verts * 16, 64);
      if (gs->gs.rec_stride)
         gs_gen = panvk_cmd_alloc_dev_mem(cmdbuf, desc, out_verts * gs->gs.rec_stride, 64);
      gs_idx = panvk_cmd_alloc_dev_mem(cmdbuf, desc, out_indices * 4, 4);
      if (!gs_pos.gpu || !gs_idx.gpu || (gs->gs.rec_stride && !gs_gen.gpu))
         return;
   }
   struct pan_ptr setup = {0}, tsetup = {0};
   if (indirect) {
      setup = panvk_cmd_alloc_dev_mem(cmdbuf, desc, sizeof(struct libpan_gs_indirect), 64);
      tsetup = panvk_cmd_alloc_dev_mem(cmdbuf, desc, sizeof(struct libpan_tess_indirect), 64);
      if (!setup.gpu || !tsetup.gpu)
         return;
   }
   const uint64_t vs_offsets = gs_upload_table(cmdbuf, vs->gs.rec_offsets);
   const uint64_t tes_offsets = gs_upload_table(cmdbuf, tes->gs.rec_offsets);
   const uint64_t gs_offsets = gs ? gs_upload_table(cmdbuf, gs->gs.rec_offsets) : 0;
   if (!blob.gpu || !in_pos.gpu || (vs->gs.rec_stride && !in_gen.gpu) || !tes_pos.gpu ||
       (tes->gs.rec_stride && !tes_gen.gpu) || !tes_idx.gpu || !vs_offsets || !tes_offsets ||
       (gs && !gs_offsets))
      return;

   /* Transform feedback from the last stage before the rasterizer, as in gs_draw. */
   const struct panvk_shader_variant *cap_stage = gs ? gs : tes;
   const uint32_t cap_invocations = gs ? gs_invocations : cap;
   cmdbuf->state.gfx.gs.xfb_counts = 0;
   cmdbuf->state.gfx.gs.xfb_staging = 0;
   cmdbuf->state.gfx.gs.xfb_invocations = 0;
   cmdbuf->state.gfx.gs.xfb_stream_index = 0;
   cmdbuf->state.gfx.gs.xfb_stream_slots = 0;
   cmdbuf->state.gfx.gs.xfb_stream_idx = 0;
   if (cmdbuf->state.gfx.xfb.active && cap_stage->gs.xfb_dwords) {
      const unsigned nstreams = gs ? MAX2(util_last_bit(gs->gs.stream_mask), 1) : 1;
      struct pan_ptr counts =
         panvk_cmd_alloc_dev_mem(cmdbuf, desc, nstreams * cap_invocations * 4, 4);
      struct pan_ptr staging = panvk_cmd_alloc_dev_mem(
         cmdbuf, desc, nstreams * out_verts * cap_stage->gs.xfb_dwords * 4, 4);
      struct pan_ptr stream_index =
         nstreams > 1
            ? panvk_cmd_alloc_dev_mem(cmdbuf, desc, (nstreams - 1) * out_indices * 4, 4)
            : (struct pan_ptr){0};
      if (!counts.gpu || !staging.gpu || (nstreams > 1 && !stream_index.gpu))
         return;
      cmdbuf->state.gfx.gs.xfb_counts = counts.gpu;
      cmdbuf->state.gfx.gs.xfb_staging = staging.gpu;
      cmdbuf->state.gfx.gs.xfb_invocations = cap_invocations;
      cmdbuf->state.gfx.gs.xfb_stream_index = stream_index.gpu;
      cmdbuf->state.gfx.gs.xfb_stream_slots = out_verts;
      cmdbuf->state.gfx.gs.xfb_stream_idx = out_indices;
   }

   /* The vertex shader: descriptors, sysvals and push uniforms as for any draw. */
   cmdbuf->state.gfx.sysvals.gs_io.pos = in_pos.gpu;
   cmdbuf->state.gfx.sysvals.gs_io.general = in_gen.gpu;
   cmdbuf->state.gfx.sysvals.gs_io.instance_stride = recs;

   const uint32_t vpp = gs ? gs->gs.output_verts_per_prim : tess_vpp;
   const uint32_t n_idx = gs ? out_indices : cap - cap % tess_vpp;
   const struct panvk_draw_info rast_draw = {
      .index.buffer_dev_addr = gs ? gs_idx.gpu : tes_idx.gpu,
      .index.buffer_size = n_idx * 4,
      .index.index_size = 4,
      .vertex.count = n_idx,
      .instance.count = 1,
      .prim = vpp == 3 ? MESA_PRIM_TRIANGLES : vpp == 2 ? MESA_PRIM_LINES : MESA_PRIM_POINTS,
   };

   struct v9_draw_state st;
   if (prepare_draw(cmdbuf, draw, &st) != VK_SUCCESS)
      return;
   build_dcd_flags(cmdbuf, panvk_shader_only_variant(get_fs(cmdbuf)), rast_draw.prim,
                   &st.flags_0, &st.flags_1);

   /* The batch's heap header, now that prepare_draw has settled the batch. */
   struct panvk_batch *batch = cmdbuf->cur_batch;
   const bool reset_heap = !batch->tess_heap;
   batch->order |= PANVK_BATCH_WAIT_FRAG | PANVK_BATCH_HOLD_FRAG;
   if (!batch->tess_heap) {
      struct pan_ptr h = panvk_cmd_alloc_dev_mem(cmdbuf, desc, sizeof(struct poly_heap), 16);
      if (!h.gpu)
         return;
      struct poly_heap *heap = h.cpu;
      heap->base = dev->tess_heap->addr.dev;
      heap->bottom = 0;
      heap->size = PANVK_TESS_HEAP_SIZE;
      batch->tess_heap = h.gpu;
   }

   const uint64_t params = blob.gpu + params_off;
   const uint64_t out = blob.gpu + out_off;
   struct poly_tess_params *p = (struct poly_tess_params *)((uint8_t *)blob.cpu + params_off);
   *p = (struct poly_tess_params){
      .heap = batch->tess_heap,
      .patch_coord_buffer = dev->tess_heap->addr.dev,
      .coord_allocs = blob.gpu + coord_off,
      .out_draws = blob.gpu + draws_off,
      .tcs_buffer = blob.gpu,
      .counts = blob.gpu + counts_off,
      .tcs_per_vertex_outputs = tcs->gs.tcs_per_vertex_outputs,
      .input_patch_size = pcp,
      .output_patch_size = tcs->gs.tcs_vertices_out,
      .tcs_patch_constants = tcs->gs.tcs_nr_patch_outputs,
      .patches_per_instance = indirect ? 0 : ppi,
      .tcs_stride_el = tcs_stride_el,
      .nr_patches = indirect ? 0 : nr_patches,
      .partitioning = spacing == TESS_SPACING_EQUAL ? POLY_TESS_PARTITIONING_INTEGER
                      : spacing == TESS_SPACING_FRACTIONAL_ODD
                         ? POLY_TESS_PARTITIONING_FRACTIONAL_ODD
                         : POLY_TESS_PARTITIONING_FRACTIONAL_EVEN,
      .points_mode = points,
      .isolines = prim == TESS_PRIMITIVE_ISOLINES,
   };
   if (!points && prim != TESS_PRIMITIVE_ISOLINES)
      p->ccw = ccw ^ (dyn->ts.domain_origin == VK_TESSELLATION_DOMAIN_ORIGIN_LOWER_LEFT);

   if (indirect) {
      struct libpan_gs_indirect *d = setup.cpu;
      *d = (struct libpan_gs_indirect){
         .pos_desc = in_pos.gpu,
         .gen_desc = in_gen.gpu,
         .cmd = draw->indirect.buffer_dev_addr,
         .fau = st.vs_fau,
         .indexed = indexed,
         .index_records = indexed ? gs_index_records(cmdbuf, draw) : 0,
         .max_vertices = max_vertices,
         .records = recs * inst,
         .first_vertex_off =
            shader_uses_sysval(vs, graphics, vs.first_vertex)
               ? shader_remapped_sysval_offset(vs, sysval_offset(graphics, vs.first_vertex))
               : ~0u,
         .base_instance_off =
            shader_uses_sysval(vs, graphics, vs.base_instance)
               ? shader_remapped_sysval_offset(vs, sysval_offset(graphics, vs.base_instance))
               : ~0u,
         .stride_off =
            shader_uses_sysval(vs, graphics, gs_io.instance_stride)
               ? shader_remapped_sysval_offset(vs,
                                               sysval_offset(graphics, gs_io.instance_stride))
               : ~0u,
      };
   }

   const struct panvk_gs_push base = {
      .in_pos = in_pos.gpu,
      .in_general = in_gen.gpu,
      .in_offsets = vs_offsets,
      .in_general_stride = vs->gs.rec_stride,
      .viewport_scale = {cmdbuf->state.gfx.sysvals.viewport.scale.x,
                         cmdbuf->state.gfx.sysvals.viewport.scale.y,
                         cmdbuf->state.gfx.sysvals.viewport.scale.z},
      .viewport_offset = {cmdbuf->state.gfx.sysvals.viewport.offset.x,
                          cmdbuf->state.gfx.sysvals.viewport.offset.y,
                          cmdbuf->state.gfx.sysvals.viewport.offset.z},
      .viewports = cmdbuf->state.gfx.gs.viewports,
      .tess_params = params,
      .out_clip_space = 1,
   };

   /* Tess control: one workgroup per (patch, instance). */
   struct panvk_compute_sysvals tcs_sv = {0};
   tcs_sv.num_work_groups.x = ppi;
   tcs_sv.num_work_groups.y = inst;
   tcs_sv.num_work_groups.z = 1;
   tcs_sv.local_group_size.x = MAX2(tcs->gs.tcs_vertices_out, 1);
   tcs_sv.local_group_size.y = 1;
   tcs_sv.local_group_size.z = 1;
   tcs_sv.gs = base;
   tcs_sv.gs.index_buf =
      indexed ? draw->index.buffer_dev_addr + (uint64_t)draw->index.offset * draw->index.index_size
              : 0;
   tcs_sv.gs.index_size = draw->index.index_size;
   tcs_sv.gs.prims_per_instance = ppi;
   tcs_sv.gs.instance_stride = inst > 1 ? recs : 0;
   tcs_sv.gs.patch_size = pcp;
   if (indirect) {
      /* Read as gs_draw's geometry shader reads an indirect draw. */
      tcs_sv.gs.indirect_cmd = setup.gpu + offsetof(struct libpan_gs_indirect, out_cmd);
      tcs_sv.gs.vertex_dcd = setup.gpu + offsetof(struct libpan_gs_indirect, dcd);
      tcs_sv.gs.index_min = setup.gpu + offsetof(struct libpan_gs_indirect, zero);
      tcs_sv.gs.in_pos_desc = setup.gpu + offsetof(struct libpan_gs_indirect, pos_desc);
      tcs_sv.gs.in_general_desc = setup.gpu + offsetof(struct libpan_gs_indirect, gen_desc);
   }

   /* Tess eval: one invocation per tessellator index, the count at out[0]. */
   const uint32_t tes_wg = DIV_ROUND_UP(cap, PANVK_GS_WORKGROUP_SIZE);
   struct panvk_compute_sysvals tes_sv = {0};
   tes_sv.num_work_groups.x = tes_wg;
   tes_sv.num_work_groups.y = 1;
   tes_sv.num_work_groups.z = 1;
   tes_sv.local_group_size.x = PANVK_GS_WORKGROUP_SIZE;
   tes_sv.local_group_size.y = 1;
   tes_sv.local_group_size.z = 1;
   tes_sv.gs = base;
   tes_sv.gs.out_pos = tes_pos.gpu;
   tes_sv.gs.out_general = tes_gen.gpu;
   tes_sv.gs.out_index = tes_idx.gpu;
   tes_sv.gs.out_offsets = tes_offsets;
   tes_sv.gs.out_general_stride = tes->gs.rec_stride;
   tes_sv.gs.num_input_prims = cap;
   tes_sv.gs.topology = PANVK_GS_TOPO_LIST;
   tes_sv.gs.prims_per_instance = cap;
   tes_sv.gs.num_prims_addr = out;

   /* The geometry shader reads the tess eval records as a list of the tessellator's primitives,
    * as many as out[2] says. */
   const uint32_t gs_wg = DIV_ROUND_UP(gs_invocations, PANVK_GS_WORKGROUP_SIZE);
   struct panvk_compute_sysvals gs_sv = {0};
   if (gs) {
      gs_sv.num_work_groups.x = gs_wg;
      gs_sv.num_work_groups.y = 1;
      gs_sv.num_work_groups.z = 1;
      gs_sv.local_group_size.x = PANVK_GS_WORKGROUP_SIZE;
      gs_sv.local_group_size.y = 1;
      gs_sv.local_group_size.z = 1;
      gs_sv.gs = base;
      gs_sv.gs.tess_params = 0;
      gs_sv.gs.in_pos = tes_pos.gpu;
      gs_sv.gs.in_general = tes_gen.gpu;
      gs_sv.gs.in_offsets = tes_offsets;
      gs_sv.gs.in_general_stride = tes->gs.rec_stride;
      gs_sv.gs.out_pos = gs_pos.gpu;
      gs_sv.gs.out_general = gs_gen.gpu;
      gs_sv.gs.out_index = gs_idx.gpu;
      gs_sv.gs.out_offsets = gs_offsets;
      gs_sv.gs.out_general_stride = gs->gs.rec_stride;
      gs_sv.gs.num_input_prims = gs_prims;
      gs_sv.gs.topology = PANVK_GS_TOPO_LIST;
      gs_sv.gs.prims_per_instance = gs_prims;
      gs_sv.gs.num_prims_addr = out + 8;
      gs_sv.gs.polygon_cull =
         ((rs->cull_mode & VK_CULL_MODE_FRONT_BIT) ? PANVK_GS_CULL_FRONT : 0) |
         ((rs->cull_mode & VK_CULL_MODE_BACK_BIT) ? PANVK_GS_CULL_BACK : 0) |
         (rs->front_face == VK_FRONT_FACE_COUNTER_CLOCKWISE ? PANVK_GS_FRONT_CCW : 0);
   }

   /* The stage that captures: the tess eval shader, or the geometry shader after it. */
   struct panvk_gs_push *cap_push = gs ? &gs_sv.gs : &tes_sv.gs;
   cap_push->xfb_counts = cmdbuf->state.gfx.gs.xfb_counts;
   cap_push->xfb_staging = cmdbuf->state.gfx.gs.xfb_staging;
   cap_push->xfb_invocations = cmdbuf->state.gfx.gs.xfb_invocations;
   cap_push->xfb_stream_index = cmdbuf->state.gfx.gs.xfb_stream_index;
   cap_push->xfb_stream_slots = cmdbuf->state.gfx.gs.xfb_stream_slots;
   cap_push->xfb_stream_idx = cmdbuf->state.gfx.gs.xfb_stream_idx;

   uint64_t tcs_rt, tcs_fau, tes_rt, tes_fau, gs_rt = 0, gs_fau = 0;
   if (!emu_prepare(cmdbuf, tcs_shader, tcs, &tcs_sv, &tcs_rt, &tcs_fau) ||
       !emu_prepare(cmdbuf, tes_shader, tes, &tes_sv, &tes_rt, &tes_fau) ||
       (gs && !emu_prepare(cmdbuf, gs_shader, gs, &gs_sv, &gs_rt, &gs_fau)))
      return;

   if (indirect) {
      struct libpan_tess_indirect *t = tsetup.cpu;
      *t = (struct libpan_tess_indirect){
         .params = params,
         .gs = setup.gpu,
         .fau = tcs_fau,
         .patch_size = pcp,
         .max_patches = PANVK_TESS_INDIRECT_GRID,
         .wg_x_off =
            shader_uses_sysval(tcs, compute, num_work_groups.x)
               ? shader_remapped_sysval_offset(tcs, sysval_offset(compute, num_work_groups.x))
               : ~0u,
         .wg_y_off =
            shader_uses_sysval(tcs, compute, num_work_groups.y)
               ? shader_remapped_sysval_offset(tcs, sysval_offset(compute, num_work_groups.y))
               : ~0u,
      };
   }

   struct panvk_precomp_ctx precomp_ctx = panvk_per_arch(precomp_cs)(cmdbuf);
#define TESS_DEPS(_x) panlib_1d_with_jm_deps(_x, batch->vtc_jc.job_index, 0)

   /* The vertex pass, sized on the GPU for an indirect draw. */
   unsigned dep = 0, tcs_setup = 0;
   struct pan_ptr vjob = panvk_cmd_alloc_desc(cmdbuf, COMPUTE_JOB);
   if (!vjob.gpu)
      return;
   if (indirect) {
      dep = emit_gs_indirect_setup(cmdbuf, setup.gpu);
      const struct panlib_tess_indirect_setup_args t_args = {.t = tsetup.gpu};
      panlib_tess_indirect_setup_struct(&precomp_ctx, TESS_DEPS(1), PANLIB_BARRIER_JM_BARRIER,
                                        t_args);
      tcs_setup = batch->vtc_jc.job_index;
      dep = emit_gs_job_patch(cmdbuf, setup.gpu, vjob.gpu, draw->indirect.buffer_dev_addr,
                              indexed, tcs_setup);
      if (!dep)
         return;
   }
   pan_section_pack(vjob.cpu, COMPUTE_JOB, PAYLOAD, cfg) {
      cfg.workgroup_size_x = 1;
      cfg.workgroup_size_y = 1;
      cfg.workgroup_size_z = 1;
      cfg.workgroup_count_x = recs;
      cfg.workgroup_count_y = inst;
      cfg.workgroup_count_z = 1;
      cfg.allow_merging_workgroups = true;
      cfg.task_increment = 1;
      cfg.task_axis = MALI_TASK_AXIS_Z;
      cfg.compute.attribute_offset = draw->vertex.base;
      cfg.compute.resources = st.vs_res;
      cfg.compute.shader = panvk_priv_mem_dev_addr(vs->spds.pos_points);
      cfg.compute.thread_storage = batch->tls.gpu;
      cfg.compute.fau = st.vs_fau;
      cfg.compute.fau_count = vs->fau.total_count;
   }
   util_dynarray_append(&batch->jobs, vjob.cpu);
   dep = pan_jc_add_job(&batch->vtc_jc, MALI_JOB_TYPE_COMPUTE, true, indirect, dep, 0, &vjob,
                        false);

   /* Tess control, its grid patched for an indirect draw. */
   struct pan_ptr tcs_job;
   const unsigned tcs_dims_dep = dep;
   if (indirect) {
      /* The job is allocated before its patch kernel so the kernel can name it. */
      tcs_job = panvk_cmd_alloc_desc(cmdbuf, COMPUTE_JOB);
      if (!tcs_job.gpu)
         return;
      dep = emit_gs_job_patch(cmdbuf, tsetup.gpu + offsetof(struct libpan_tess_indirect, tcs),
                              tcs_job.gpu, 0, false, tcs_dims_dep);
      if (!dep)
         return;
      struct pan_compute_dim dim = {PANVK_TESS_INDIRECT_GRID, 1, 1};
      const uint64_t tls = panvk_per_arch(cmd_dispatch_prepare_tls)(cmdbuf, tcs, &dim, true);
      panvk_per_arch(jm_pack_compute_payload)(tcs_job.cpu, tcs, 1, 1, 1, tcs_rt, tcs_fau, tls,
                                              tcs->fau.total_count);
      util_dynarray_append(&batch->jobs, tcs_job.cpu);
      dep = pan_jc_add_job(&batch->vtc_jc, MALI_JOB_TYPE_COMPUTE, true, true, dep, 0, &tcs_job,
                           false);
   } else {
      dep = emu_job(cmdbuf, tcs, ppi, inst, false, tcs_rt, tcs_fau, dep, NULL);
   }
   if (!dep)
      return;

   for (unsigned pass = 0; pass < 2; pass++) {
      const uint32_t mode = pass == 0 ? POLY_TESS_MODE_COUNT : POLY_TESS_MODE_WITH_COUNTS;
      if (prim == TESS_PRIMITIVE_ISOLINES) {
         const struct panlib_tess_isoline_args a = {.p = params, .mode = mode};
         panlib_tess_isoline_struct(&precomp_ctx, TESS_DEPS(nr_patches),
                                    PANLIB_BARRIER_JM_BARRIER, a);
      } else if (prim == TESS_PRIMITIVE_QUADS) {
         const struct panlib_tess_quad_args a = {.p = params, .mode = mode};
         panlib_tess_quad_struct(&precomp_ctx, TESS_DEPS(nr_patches), PANLIB_BARRIER_JM_BARRIER,
                                 a);
      } else {
         const struct panlib_tess_tri_args a = {.p = params, .mode = mode};
         panlib_tess_tri_struct(&precomp_ctx, TESS_DEPS(nr_patches), PANLIB_BARRIER_JM_BARRIER,
                                a);
      }
      if (pass == 0) {
         const struct panlib_tess_prefix_sum_args a = {
            .p = params, .reset_heap = reset_heap, .out = out};
         panlib_tess_prefix_sum_struct(&precomp_ctx, TESS_DEPS(1), PANLIB_BARRIER_JM_BARRIER, a);
      }
   }
#undef TESS_DEPS

   dep = emu_job(cmdbuf, tes, tes_wg, 1, false, tes_rt, tes_fau, batch->vtc_jc.job_index, NULL);
   if (dep && gs)
      dep = emu_job(cmdbuf, gs, gs_wg, 1, false, gs_rt, gs_fau, dep, NULL);
   if (!dep)
      return;
   if (cmdbuf->state.gfx.gs.xfb_counts)
      emit_xfb_jobs(cmdbuf, cap_stage, gs ? gs_idx.gpu : tes_idx.gpu, dep, gs ? 0 : tess_vpp);

   /* The rasterization pass. The tess eval shader leaves one vertex per tessellator index in
    * index order, and the identity index buffer it writes draws the tessellator's own
    * primitives; slots past the real count stay degenerate. */
   cmdbuf->state.gfx.sysvals.gs_io.pos = gs ? gs_pos.gpu : tes_pos.gpu;
   cmdbuf->state.gfx.sysvals.gs_io.general = gs ? gs_gen.gpu : tes_gen.gpu;
   struct pan_ptr rast_pu;
   if (panvk_per_arch(cmd_prepare_gfx_push_uniforms)(cmdbuf, rast, &rast_pu, 1) != VK_SUCCESS)
      return;
   st.vs = rast;
   st.vs_res = 0;
   st.vs_fau = rast_pu.gpu;

   uint32_t view_mask = cmdbuf->state.gfx.render.view_mask;
   uint32_t layer_count = view_mask ? util_bitcount(view_mask)
                                    : cmdbuf->state.gfx.render.layer_count;
   for (uint32_t i = 0; i < layer_count; i++) {
      const uint32_t layer = view_mask ? u_bit_scan(&view_mask) : i;
      struct pan_ptr job;
      if (emit_draw_job(cmdbuf, &rast_draw, &st, layer, &job) != VK_SUCCESS)
         return;
      pan_jc_add_job(&batch->vtc_jc, MALI_JOB_TYPE_MALLOC_VERTEX, true, false, dep, 0, &job,
                     false);
   }

   cmdbuf->state.v9.valid = false;
   clear_dirty_after_draw(cmdbuf);
}

static void
panvk_cmd_draw(struct panvk_cmd_buffer *cmdbuf, const struct panvk_draw_info *draw,
               uint64_t indirect_cmd)
{
   const struct panvk_shader_variant *vs =
      panvk_shader_hw_variant(cmdbuf->state.gfx.vs.shader);

   /* If there's no vertex shader, we can skip the draw. */
   if (!vs || !panvk_priv_mem_check_alloc(vs->spd))
      return;

   if (!indirect_cmd)
      panvk_stats_add_draw(&cmdbuf->state.gfx, &cmdbuf->vk.dynamic_graphics_state, draw);

   cmdbuf->state.gfx.fs.required =
      fs_required(&cmdbuf->state.gfx, &cmdbuf->vk.dynamic_graphics_state);

   /* Rasterizer discard without an occlusion query draws nothing, unless a geometry shader
    * still has to run for transform feedback. */
   if (cmdbuf->vk.dynamic_graphics_state.rs.rasterizer_discard_enable &&
       cmdbuf->state.gfx.occlusion_query.mode == MALI_OCCLUSION_MODE_DISABLED &&
       !((cmdbuf->state.gfx.gs.app_shader || cmdbuf->state.gfx.tess.tes ||
          vs_xfb_draw(cmdbuf)) &&
         cmdbuf->state.gfx.xfb.active))
      return;

   /* An indirect draw through the emulation carries its command in draw->indirect. */
   struct panvk_draw_info emu = *draw;
   emu.indirect.buffer_dev_addr = indirect_cmd;
   emu.indirect.draw_count = indirect_cmd ? 1 : 0;

   if (cmdbuf->state.gfx.tess.tes) {
      cmdbuf->state.v9.valid = false;
      gfx_state_set_dirty(cmdbuf, VS_PUSH_UNIFORMS);
      tess_draw(cmdbuf, &emu);
      return;
   }

   const enum gs_emulation mode =
      cmdbuf->state.gfx.gs.app_shader   ? GS_EMULATION_APP
      : vs_xfb_draw(cmdbuf)             ? GS_EMULATION_XFB
      : polygon_mode_draw(cmdbuf, draw) ? GS_EMULATION_POLYGON_MODE
                                        : GS_EMULATION_NONE;
   if (mode != GS_EMULATION_NONE) {
      cmdbuf->state.v9.valid = false;
      gfx_state_set_dirty(cmdbuf, VS_PUSH_UNIFORMS);
      gs_draw(cmdbuf, &emu, mode);
      return;
   }

   struct v9_draw_state st;
   if (prepare_draw(cmdbuf, draw, &st) != VK_SUCCESS)
      return;

   uint32_t view_mask = cmdbuf->state.gfx.render.view_mask;
   uint32_t layer_count = view_mask ? util_bitcount(view_mask)
                                    : cmdbuf->state.gfx.render.layer_count;

   for (uint32_t i = 0; i < layer_count; i++) {
      const uint32_t layer = view_mask ? u_bit_scan(&view_mask) : i;
      struct pan_ptr job;
      if (emit_draw_job(cmdbuf, draw, &st, layer, &job) != VK_SUCCESS)
         return;

      if (indirect_cmd) {
         if (patch_indirect_job(cmdbuf, draw, indirect_cmd, job) != VK_SUCCESS)
            return;
      } else {
         /* Not chained to the previous draw's job: the hardware keeps the draws of a chain in
          * order on its own (probe/mali/bench BENCH_ORDER=1..3 hash the same either way, and as
          * the vendor driver's), and waiting for each job to finish before the next one starts
          * made 2000 small draws take 5.9 ms on Mali-G77 instead of 2.1. */
         struct pan_jc *jc = &cmdbuf->cur_batch->vtc_jc;
         jc->tiler_dep = 0;
         pan_jc_add_job(jc, MALI_JOB_TYPE_MALLOC_VERTEX, false, false, 0, 0, &job, false);
      }
   }

   /* The indirect kernel patches the sysvals and the per-instance attributes in place. */
   if (indirect_cmd)
      cmdbuf->state.v9.valid = false;
   clear_dirty_after_draw(cmdbuf);
}

void
panvk_per_arch(cmd_finish_draw_helper)(struct panvk_cmd_buffer *cmdbuf,
                                       struct panvk_batch *batch)
{
   /* No draw helper on v9: indirect draws patch their own job (patch_indirect_job). */
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdDraw)(VkCommandBuffer commandBuffer, uint32_t vertexCount,
                        uint32_t instanceCount, uint32_t firstVertex,
                        uint32_t firstInstance)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);

   if (instanceCount == 0 || vertexCount == 0)
      return;

   struct panvk_draw_info draw = {
      .vertex.base = firstVertex,
      .vertex.count = vertexCount,
      .instance.base = firstInstance,
      .instance.count = instanceCount,
      .prim = panvk_get_client_prim(cmdbuf),
   };

   panvk_cmd_draw(cmdbuf, &draw, 0);
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdDrawIndexed)(VkCommandBuffer commandBuffer,
                               uint32_t indexCount, uint32_t instanceCount,
                               uint32_t firstIndex, int32_t vertexOffset,
                               uint32_t firstInstance)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);

   if (instanceCount == 0 || indexCount == 0)
      return;

   struct panvk_draw_info draw = {
      .index = panvk_draw_info_index(cmdbuf, firstIndex),
      .vertex.base = vertexOffset,
      .vertex.count = indexCount,
      .instance.base = firstInstance,
      .instance.count = instanceCount,
      .prim = panvk_get_client_prim(cmdbuf),
   };

   panvk_cmd_draw(cmdbuf, &draw, 0);
}

/* One job per draw of the indirect buffer, each with its own descriptors and push uniforms. */
static void
cmd_draw_indirect(struct panvk_cmd_buffer *cmdbuf, VkBuffer _buffer, VkDeviceSize offset,
                  uint32_t draw_count, uint32_t stride, bool indexed)
{
   VK_FROM_HANDLE(panvk_buffer, buffer, _buffer);

   for (uint32_t i = 0; i < draw_count; i++) {
      struct panvk_draw_info draw = {
         .vertex.count = 1,
         .instance.count = 1,
         .prim = panvk_get_client_prim(cmdbuf),
      };
      if (indexed) {
         const __typeof__(draw.index) index = panvk_draw_info_index(cmdbuf, 0);
         draw.index = index;
      }

      /* Nothing of the previous draw is reused: its job was patched on the GPU. */
      cmdbuf->state.v9.valid = false;
      gfx_state_set_dirty(cmdbuf, VS_PUSH_UNIFORMS);
      panvk_cmd_draw(cmdbuf, &draw,
                     panvk_buffer_gpu_ptr(buffer, offset + (uint64_t)i * stride));
   }
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdDrawIndirect)(VkCommandBuffer commandBuffer, VkBuffer _buffer,
                                VkDeviceSize offset, uint32_t drawCount,
                                uint32_t stride)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);
   cmd_draw_indirect(cmdbuf, _buffer, offset, drawCount, stride, false);
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdDrawIndexedIndirect)(VkCommandBuffer commandBuffer,
                                       VkBuffer _buffer, VkDeviceSize offset,
                                       uint32_t drawCount, uint32_t stride)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);
   cmd_draw_indirect(cmdbuf, _buffer, offset, drawCount, stride, true);
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdBeginRendering)(VkCommandBuffer commandBuffer,
                                  const VkRenderingInfo *pRenderingInfo)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);
   struct panvk_cmd_graphics_state *state = &cmdbuf->state.gfx;
   bool resuming = pRenderingInfo->flags & VK_RENDERING_RESUMING_BIT;

   /* New attachments: the blend and depth/stencil descriptors depend on them. */
   cmdbuf->state.v9.valid = false;

   if (resuming && cmdbuf->cur_batch) {
      state->render.flags = pRenderingInfo->flags;
   } else {
      if (cmdbuf->cur_batch)
         panvk_per_arch(cmd_close_batch)(cmdbuf);

      panvk_per_arch(cmd_init_render_state)(cmdbuf, pRenderingInfo);
      cmdbuf->state.gfx.render.fb.needs_load = !resuming;
   }

   if (!cmdbuf->cur_batch)
      panvk_per_arch(cmd_open_batch)(cmdbuf);
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdEndRendering)(VkCommandBuffer commandBuffer)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);

   if (!(cmdbuf->state.gfx.render.flags & VK_RENDERING_SUSPENDING_BIT)) {
      const struct pan_fb_load *fb_load = &cmdbuf->state.gfx.render.fb.load;
      bool always_load = fb_load->z.always || fb_load->s.always;
      for (unsigned rt = 0; rt < PAN_MAX_RTS; rt++) {
         if (fb_load->rts[rt].always)
            always_load = true;
      }

      if (always_load)
         panvk_per_arch(cmd_alloc_fb_desc)(cmdbuf);

      cmdbuf->state.gfx.render.fb.needs_store = true;

      panvk_per_arch(cmd_close_batch)(cmdbuf);
      cmdbuf->cur_batch = NULL;
      panvk_per_arch(cmd_meta_resolve_attachments)(cmdbuf);
   }
}

/* --- VK_EXT_transform_feedback (captured from the geometry shader emulation) ----------------- */

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdBindTransformFeedbackBuffers2EXT)(
   VkCommandBuffer commandBuffer, uint32_t firstBinding, uint32_t bindingCount,
   const VkBindTransformFeedbackBuffer2InfoEXT *pBindingInfos)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);

   for (uint32_t i = 0; i < bindingCount && firstBinding + i < 4; i++) {
      cmdbuf->state.gfx.xfb.addr[firstBinding + i] = pBindingInfos[i].addressRange.address;
      cmdbuf->state.gfx.xfb.size[firstBinding + i] =
         MIN2(pBindingInfos[i].addressRange.size, UINT32_MAX);
   }
}

/* Load the write offsets from the counter buffers (Begin) or store them there (End). Without a
 * counter, Begin starts at 0, which the CPU can write itself. */
static void
xfb_counters(struct panvk_cmd_buffer *cmdbuf, uint32_t first, uint32_t count,
             const VkBindTransformFeedbackBuffer2InfoEXT *infos, bool load)
{
   uint64_t c[4] = {0};
   for (uint32_t i = 0; infos && i < count && first + i < 4; i++)
      c[first + i] = infos[i].addressRange.address;
   const bool any = c[0] || c[1] || c[2] || c[3];

   if (load) {
      struct pan_ptr slot = panvk_cmd_alloc_dev_mem(cmdbuf, desc, 16, 16);
      if (!slot.gpu)
         return;
      memset(slot.cpu, 0, 16);
      cmdbuf->state.gfx.xfb.offsets = slot.gpu;
   }
   if (!any)
      return;

   if (!cmdbuf->cur_batch)
      panvk_per_arch(cmd_open_batch)(cmdbuf);

   struct panvk_precomp_ctx precomp_ctx = panvk_per_arch(precomp_cs)(cmdbuf);
   const struct panlib_xfb_counters_args args = {
      .offsets = cmdbuf->state.gfx.xfb.offsets,
      .c0 = c[0],
      .c1 = c[1],
      .c2 = c[2],
      .c3 = c[3],
      .load = load,
   };
   panlib_xfb_counters_struct(&precomp_ctx,
                              panlib_1d_with_jm_deps(1, 0, cmdbuf->state.gfx.xfb.last_job),
                              PANLIB_BARRIER_JM_BARRIER, args);
   cmdbuf->state.gfx.xfb.last_job = cmdbuf->cur_batch->vtc_jc.job_index;
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdBeginTransformFeedback2EXT)(
   VkCommandBuffer commandBuffer, uint32_t firstCounterRange, uint32_t counterRangeCount,
   const VkBindTransformFeedbackBuffer2InfoEXT *pCounterInfos)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);

   xfb_counters(cmdbuf, firstCounterRange, counterRangeCount, pCounterInfos, true);
   cmdbuf->state.gfx.xfb.active = true;
}

VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdEndTransformFeedback2EXT)(
   VkCommandBuffer commandBuffer, uint32_t firstCounterRange, uint32_t counterRangeCount,
   const VkBindTransformFeedbackBuffer2InfoEXT *pCounterInfos)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);

   xfb_counters(cmdbuf, firstCounterRange, counterRangeCount, pCounterInfos, false);
   cmdbuf->state.gfx.xfb.active = false;
}

/* An indirect draw whose vertex count panlib_xfb_byte_count derives from the counter, at the end
 * of the current batch (after its capture jobs); the draw goes into the next one. */
VKAPI_ATTR void VKAPI_CALL
panvk_per_arch(CmdDrawIndirectByteCount2EXT)(
   VkCommandBuffer commandBuffer, uint32_t instanceCount, uint32_t firstInstance,
   const VkBindTransformFeedbackBuffer2InfoEXT *pCounterInfo, uint32_t counterOffset,
   uint32_t vertexStride)
{
   VK_FROM_HANDLE(panvk_cmd_buffer, cmdbuf, commandBuffer);

   if (!instanceCount || !vertexStride || !pCounterInfo->addressRange.address)
      return;

   struct pan_ptr c = panvk_cmd_alloc_dev_mem(cmdbuf, desc, sizeof(VkDrawIndirectCommand), 4);
   if (!c.gpu)
      return;
   *(VkDrawIndirectCommand *)c.cpu = (VkDrawIndirectCommand){
      .instanceCount = instanceCount,
      .firstInstance = firstInstance,
   };

   if (!cmdbuf->cur_batch)
      panvk_per_arch(cmd_open_batch)(cmdbuf);
   struct panvk_precomp_ctx precomp_ctx = panvk_per_arch(precomp_cs)(cmdbuf);
   const struct panlib_xfb_byte_count_args args = {
      .counter = pCounterInfo->addressRange.address,
      .cmd = c.gpu,
      .counter_offset = counterOffset,
      .vertex_stride = vertexStride,
   };
   panlib_xfb_byte_count_struct(&precomp_ctx,
                                panlib_1d_with_jm_deps(1, 0, cmdbuf->state.gfx.xfb.last_job),
                                PANLIB_BARRIER_JM_BARRIER, args);
   panvk_per_arch(cmd_close_batch)(cmdbuf);
   panvk_per_arch(cmd_open_batch)(cmdbuf);

   const struct panvk_draw_info draw = {
      .vertex.count = 1,
      .instance.count = 1,
      .prim = panvk_get_client_prim(cmdbuf),
   };
   cmdbuf->state.v9.valid = false;
   gfx_state_set_dirty(cmdbuf, VS_PUSH_UNIFORMS);
   panvk_cmd_draw(cmdbuf, &draw, c.gpu);
}
