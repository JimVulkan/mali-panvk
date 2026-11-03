/*
 * Copyright 2025 Collabora Ltd.
 * SPDX-License-Identifier: MIT
 */

#include <stdatomic.h>
#include <stdint.h>
#include "compiler/libcl/libcl.h"
#include "genxml/gen_macros.h"

#pragma once

#if (PAN_ARCH == 6 || PAN_ARCH == 7)
/* Keep in sync with panvk_varying_buf_id */
enum panlib_varying_buf_id {
   PANLIB_VARY_BUF_GENERAL,
   PANLIB_VARY_BUF_POSITION,
   PANLIB_VARY_BUF_PSIZ,

   /* Keep last */
   PANLIB_VARY_BUF_MAX,
};

struct libpan_draw_helper_index_min_max_result {
    uint32_t min;
    uint32_t max;
};

struct libpan_draw_helper_attrib_buf_info {
    uint32_t divisor;
    uint32_t stride;
    bool per_instance;
};

struct libpan_draw_helper_attrib_info {
    uint32_t base_offset;

    /* When the attribute is per instance, the stride otherwise 0 */
    uint32_t stride;
};

/* One indirect draw (or indexed draw rerouted through the indirect path) of a batch. The CPU
 * writes one per draw and layer; panlib_draw_batch_minmax and panlib_draw_batch_patch run once
 * for all of them. Pointers are GPU addresses, 0 when absent. */
struct libpan_draw_batch_entry {
    uint64_t cmd;                  /* VkDraw(Indexed)IndirectCommand */
    uint64_t index_buffer_ptr;
    uint64_t index_min_out;        /* uint32_t: the index minimum, for the GS emulation */
    uint64_t varying_bufs_descs;
    uint64_t heap;                 /* struct poly_heap: the varyings of every draw */
    uint64_t attrib_bufs_descs;
    uint64_t attrib_bufs_infos;
    uint64_t attribs_descs;
    uint64_t attribs_infos;
    uint64_t first_vertex_sysval;
    uint64_t first_instance_sysval;
    uint64_t raw_vertex_offset_sysval;
    uint64_t idvs_job;
    uint64_t vertex_job;
    uint64_t tiler_job;
    uint32_t index_size;           /* 0: not indexed */
    uint32_t primitive_restart;
    uint32_t primitive_vertex_count;
    uint32_t attrib_bufs_valid;
    uint32_t attribs_valid;
    uint32_t index_min;            /* scan result: UINT32_MAX and 0 before the scan and after */
    uint32_t index_max;            /* the patch, so the command buffer can run again */
    uint32_t pad;
};

struct libpan_draw_batch {
    uint64_t entries;              /* struct libpan_draw_batch_entry[count] */
    uint32_t count;
    uint32_t pad;
};

#endif

#if PAN_ARCH >= 6
/* Transform feedback captured from the geometry shader emulation, one per capturing draw. The draw
 * path fills everything above `start`; panlib_xfb_scan fills the rest, panlib_xfb_copy reads it. */
struct libpan_xfb_draw {
    uint64_t buffer[4];      /* bound address */
    uint32_t size[4];        /* bound size in bytes */
    uint32_t stride[4];      /* the shader's xfb stride, 0 = not captured to */
    uint64_t offsets_in;     /* uint32_t[4]: write offsets before this draw */
    uint64_t offsets_out;    /* uint32_t[4]: after it */
    uint64_t counts;         /* uint32_t per emulation invocation: primitives it emitted */
    uint64_t first;          /* uint32_t per emulation invocation: its first global primitive */
    uint64_t indices;        /* the emulation's index buffer */
    uint64_t staging;        /* record_dwords per output vertex slot */
    uint64_t table;          /* uint32_t per record dword: buffer << 16 | dword in the vertex */
    uint32_t invocations;    /* emulation invocations dispatched */
    uint32_t idx_per_invocation;
    uint32_t vpp;            /* vertices per output primitive */
    uint32_t record_dwords;
    uint32_t start[4];       /* scan: byte offset this draw writes each buffer from */
    uint32_t written;        /* scan: primitives captured */
    uint32_t needed;         /* scan: primitives emitted */
    uint32_t count_stride;   /* invocation i reads counts[i * count_stride]; 0 means 1 */
    uint32_t pad;
    uint64_t stats;          /* uint64_t[2] the scan adds written and needed to, or 0 */
};
#endif

#if PAN_ARCH >= 9
/* One indirect draw through the geometry shader emulation, sized by panlib_gs_indirect_setup.
 * The draw path fills everything above job_x; the emulation reads out_cmd as its indirect
 * command, dcd as its vertex job's DRAW word, zero as index_min and pos_desc/gen_desc as the
 * varying buffer descriptors (see struct panvk_gs_push). */
struct libpan_gs_indirect {
   uint64_t pos_desc;          /* the vertex records, as an ATTRIBUTE_BUFFER pointer */
   uint64_t gen_desc;
   uint64_t cmd;               /* VkDraw[Indexed]IndirectCommand */
   uint64_t count;             /* the draw count buffer, or 0 */
   uint64_t fau;               /* the vertex shader's FAU */
   uint32_t draw;              /* this draw, against *count */
   uint32_t indexed;
   uint32_t index_records;     /* indexed: records per instance */
   uint32_t max_vertices;      /* non-indexed: vertexCount cap */
   uint32_t records;           /* allocated records */
   uint32_t first_vertex_off;  /* FAU byte offsets, ~0 if the shader does not read it */
   uint32_t base_instance_off;
   uint32_t stride_off;
   uint32_t job_x;             /* the vertex job: x, y and the attribute offset */
   uint32_t job_y;
   uint32_t attrib_offset;
   uint32_t dcd;               /* the padded instance stride in bits 16-23 */
   uint32_t out_cmd[5];
   uint32_t zero;
};

/* An indirect tessellated draw, after panlib_gs_indirect_setup has run for its vertex pass:
 * panlib_tess_indirect_setup turns the clamped command into patch counts in the tess params and
 * the tess control pass's grid. */
struct libpan_tess_indirect {
   uint64_t params;            /* struct poly_tess_params */
   uint64_t gs;                /* the vertex pass's struct libpan_gs_indirect */
   uint64_t fau;               /* the tess control shader's FAU */
   uint32_t patch_size;
   uint32_t max_patches;       /* the patch arrays' size; a draw with more is dropped */
   uint32_t wg_x_off;          /* FAU byte offsets of num_work_groups.x/y, ~0 if unread */
   uint32_t wg_y_off;
   struct libpan_gs_indirect tcs; /* only job_x, job_y and attrib_offset: the tess control grid */
};
#endif

#if PAN_ARCH == 9
/* One indirect draw on v9, patched into its MALLOC_VERTEX job by panlib_v9_draw_indirect. */
struct libpan_v9_indirect_draw {
   uint64_t cmd;                  /* VkDraw[Indexed]IndirectCommand */
   uint64_t job;                  /* the MALLOC_VERTEX job */
   uint64_t index_buffer;         /* 0 for a non-indexed draw */
   uint64_t first_vertex_sysval;  /* 0 if the shader does not read it */
   uint64_t base_instance_sysval; /* 0 if the shader does not read it */
   uint64_t attribs;              /* the vertex shader's ATTRIBUTE descriptors */
   uint32_t index_size;
   uint32_t patch_attribs;        /* per-instance attributes offset by firstInstance * stride */
   uint32_t strides[16];
};

/* An indirect draw through the geometry shader emulation on v9: panlib_v9_gs_job copies the
 * size panlib_gs_indirect_setup (or panlib_tess_indirect_setup) wrote into a COMPUTE job, and
 * firstInstance into the per-instance attribute offsets, as panlib_v9_draw_indirect does. */
struct libpan_v9_gs_job {
   uint64_t dims;                 /* struct libpan_gs_indirect: job_x, job_y, attrib_offset */
   uint64_t job;                  /* the COMPUTE job */
   uint64_t cmd;                  /* VkDraw[Indexed]IndirectCommand, 0: no attribute patch */
   uint64_t attribs;              /* the vertex shader's ATTRIBUTE descriptors */
   uint32_t indexed;
   uint32_t patch_attribs;
   uint32_t strides[16];
};
#endif
