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
 * Not done yet: geometry and tessellation shaders, transform feedback.
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
   const VkViewport *viewport =
      &cmdbuf->vk.dynamic_graphics_state.vp.viewports[0];
   const VkRect2D *scissor = &cmdbuf->vk.dynamic_graphics_state.vp.scissors[0];

   pan_pack((struct mali_scissor_packed *)out, SCISSOR, cfg) {
      int minx = (int)viewport->x;
      int maxx = (int)(viewport->x + viewport->width);
      int miny = MIN2((int)viewport->y, (int)(viewport->y + viewport->height));
      int maxy = MAX2((int)viewport->y, (int)(viewport->y + viewport->height));

      minx = MAX2(scissor->offset.x, minx);
      miny = MAX2(scissor->offset.y, miny);
      maxx = MIN2(scissor->offset.x + (int)scissor->extent.width, maxx);
      maxy = MIN2(scissor->offset.y + (int)scissor->extent.height, maxy);

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

   panvk_depth_range(&cmdbuf->state.gfx, &cmdbuf->vk.dynamic_graphics_state.vp,
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

   if (cmdbuf->state.gfx.ib.cpu_readable) {
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
 * jm/panvk_vX_cmd_draw.c, emit_xfb_jobs). */
static void
emit_xfb_jobs(struct panvk_cmd_buffer *cmdbuf, const struct panvk_shader_variant *gs,
              uint64_t out_index, unsigned dep_job_id)
{
   struct panvk_batch *batch = cmdbuf->cur_batch;
   const uint32_t n = cmdbuf->state.gfx.gs.xfb_invocations;
   const uint32_t k_dw = gs->gs.xfb_dwords;

   u_foreach_bit(s, gs->gs.stream_mask) {
      bool captures = false;
      for (unsigned b = 0; b < 4; b++)
         captures |= gs->gs.xfb_strides[b] && gs->gs.xfb_buffer_stream[b] == s;
      if (!captures || s)
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
         .counts = cmdbuf->state.gfx.gs.xfb_counts,
         .first = table + k_dw * 4,
         .indices = out_index,
         .staging = cmdbuf->state.gfx.gs.xfb_staging,
         .table = table,
         .invocations = n,
         .idx_per_invocation = gs->gs.max_output_prims * gs->gs.output_verts_per_prim,
         .vpp = gs->gs.output_verts_per_prim,
         .record_dwords = k_dw,
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
static void
gs_draw(struct panvk_cmd_buffer *cmdbuf, const struct panvk_draw_info *draw)
{
   const struct panvk_shader *gs_shader = cmdbuf->state.gfx.gs.app_shader;
   const struct panvk_shader_variant *gs = panvk_shader_only_variant(gs_shader);
   const struct panvk_shader_variant *rast = panvk_shader_gs_rast_variant(gs_shader);
   const struct panvk_shader_variant *vs = panvk_shader_hw_variant(cmdbuf->state.gfx.vs.shader);

   if (!vs->gs.vs_for_gs || !panvk_priv_mem_check_alloc(gs->spd) ||
       !panvk_priv_mem_check_alloc(rast->spd))
      return;

   const bool indexed = draw->index.index_size != 0;
   const uint32_t inst = draw->instance.count;
   const uint32_t recs = indexed ? gs_index_records(cmdbuf, draw) : draw->vertex.count;
   const uint32_t prims_per_instance = gs_prims_per_instance(draw, gs);
   const uint32_t in_prims = prims_per_instance * inst;
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
      gs->gs.rec_stride ? panvk_cmd_alloc_dev_mem(cmdbuf, desc, out_verts * gs->gs.rec_stride, 64)
                        : (struct pan_ptr){0};
   struct pan_ptr out_idx = panvk_cmd_alloc_dev_mem(cmdbuf, desc, out_indices * 4, 4);
   /* Transform feedback: a primitive count per emulation invocation and a capture record per
    * output vertex slot (stream 0). */
   cmdbuf->state.gfx.gs.xfb_counts = 0;
   cmdbuf->state.gfx.gs.xfb_staging = 0;
   cmdbuf->state.gfx.gs.xfb_invocations = 0;
   if (cmdbuf->state.gfx.xfb.active && gs->gs.xfb_dwords) {
      struct pan_ptr counts = panvk_cmd_alloc_dev_mem(cmdbuf, desc, invocations * 4, 4);
      struct pan_ptr staging =
         panvk_cmd_alloc_dev_mem(cmdbuf, desc, out_verts * gs->gs.xfb_dwords * 4, 4);
      if (!counts.gpu || !staging.gpu)
         return;
      cmdbuf->state.gfx.gs.xfb_counts = counts.gpu;
      cmdbuf->state.gfx.gs.xfb_staging = staging.gpu;
      cmdbuf->state.gfx.gs.xfb_invocations = invocations;
   }

   const uint64_t in_offsets = gs_upload_table(cmdbuf, vs->gs.rec_offsets);
   const uint64_t out_offsets = gs_upload_table(cmdbuf, gs->gs.rec_offsets);
   if (!in_pos.gpu || !out_pos.gpu || !out_idx.gpu || !in_offsets || !out_offsets ||
       (vs->gs.rec_stride && !in_gen.gpu) || (gs->gs.rec_stride && !out_gen.gpu))
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

   struct panvk_batch *batch = cmdbuf->cur_batch;

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
   const unsigned vjob_id =
      pan_jc_add_job(&batch->vtc_jc, MALI_JOB_TYPE_COMPUTE, false, false, 0, 0, &vjob, false);

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
      .out_general_stride = gs->gs.rec_stride,
      .num_input_prims = in_prims,
      .index_buf = indexed ? draw->index.buffer_dev_addr +
                                (uint64_t)draw->index.offset * draw->index.index_size
                           : 0,
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
      emit_xfb_jobs(cmdbuf, gs, out_idx.gpu, gjob_id);

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

static void
panvk_cmd_draw(struct panvk_cmd_buffer *cmdbuf, const struct panvk_draw_info *draw,
               uint64_t indirect_cmd)
{
   const struct panvk_shader_variant *vs =
      panvk_shader_hw_variant(cmdbuf->state.gfx.vs.shader);

   /* If there's no vertex shader, we can skip the draw. */
   if (!vs || !panvk_priv_mem_check_alloc(vs->spd))
      return;

   cmdbuf->state.gfx.fs.required =
      fs_required(&cmdbuf->state.gfx, &cmdbuf->vk.dynamic_graphics_state);

   /* Rasterizer discard without an occlusion query draws nothing, unless a geometry shader
    * still has to run for transform feedback. */
   if (cmdbuf->vk.dynamic_graphics_state.rs.rasterizer_discard_enable &&
       cmdbuf->state.gfx.occlusion_query.mode == MALI_OCCLUSION_MODE_DISABLED &&
       !(cmdbuf->state.gfx.gs.app_shader && cmdbuf->state.gfx.xfb.active))
      return;

   if (cmdbuf->state.gfx.gs.app_shader) {
      if (indirect_cmd) {
         static bool warned;
         if (!warned) {
            warned = true;
            mesa_logw("panvk v9: indirect draws with a geometry shader are not supported yet");
         }
         return;
      }
      cmdbuf->state.v9.valid = false;
      gs_draw(cmdbuf, draw);
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
