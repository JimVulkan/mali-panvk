/*
 * Copyright 2026 JimVulkan
 * SPDX-License-Identifier: MIT
 *
 * GPU decoder/transcoder for BC4, BC5, BC6H and BC7, run once at upload.
 * One invocation handles one 4x4 block; the dispatch in radv_meta_bc_decode.c
 * is sized in blocks to match.
 *
 * The BC6H/BC7 and RGTC decode follows Mesa's CPU decoders, and its tables are generated from
 * them; do not hand-edit:
 *   src/util/format/texcompress_bptc_tmp.h  Copyright (C) 2014 Intel Corporation (MIT)
 *   src/util/format/u_format_rgtc.c         Copyright (C) 2011 Red Hat Inc. (MIT)
 */
#version 460
#extension GL_EXT_samplerless_texture_functions : require
#if defined(BC_OUT_ASTC7) || defined(BC_OUT_ASTC7BUF)
#extension GL_EXT_control_flow_attributes : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#endif

/* BC_OUT_EACBUF / BC_OUT_EACRBUF / BC_OUT_BC3BUF / BC_OUT_ASTC7BUF: the BC5 -> EAC_R11G11,
 * BC4 -> EAC_R11, BC7 -> BC3 and BC7 -> ASTC transcodes reading the blocks straight from the copy's
 * source buffer (pc.src_va) instead of plane 0, so an upload is one pass. Used when the carrier plane
 * shares plane 0's memory (radv_image.xclipse_bc_alias). */
#if defined(BC_OUT_EACBUF) || defined(BC_OUT_EACRBUF) || defined(BC_OUT_BC3BUF) || defined(BC_OUT_ASTC7BUF)
#if defined(BC_OUT_EACBUF)
#define BC_OUT_EAC
#elif defined(BC_OUT_EACRBUF)
#define BC_OUT_EAC_R
#elif defined(BC_OUT_ASTC7BUF)
#define BC_OUT_ASTC7
#else
#define BC_OUT_BC3
#endif
#define BC_SRC_BUFFER
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference_uvec2 : require
layout(buffer_reference, std430, buffer_reference_align = 16) readonly buffer BcSrcBlocks { uvec4 b[]; };
/* 8-byte blocks (BC4). */
layout(buffer_reference, std430, buffer_reference_align = 8) readonly buffer BcSrcBlocks8 { uvec2 b[]; };
#endif

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

/* Compressed plane, one uint texel per 4x4 block: RG32_UINT for 8-byte blocks, RGBA32_UINT for
 * 16-byte blocks. */
/* BC_3D: the same for 3D images, whose depth slices take the place of the array layers. */
#if defined(BC_3D)
#define BC_TEXTURE utexture3D
#define BC_UIMAGE  uimage3D
#define BC_IMAGE   image3D
#else
#define BC_TEXTURE utexture2DArray
#define BC_UIMAGE  uimage2DArray
#define BC_IMAGE   image2DArray
#endif

layout(set = 0, binding = 0) uniform BC_TEXTURE s_blocks;

#if defined(BC_OUT_U32X4) || defined(BC_OUT_EAC) || defined(BC_OUT_BC3) || defined(BC_OUT_ASTC) || \
    defined(BC_OUT_ASTC7)
/* Transcode: the destination is a compressed image, one rgba32ui texel per 16-byte block.
 *   BC_OUT_U32X4 -- BC4 -> BC3 byte shuffle
 *   BC_OUT_EAC   -- BC5 -> EAC_R11G11 re-encode
 *   BC_OUT_BC3   -- BC7 -> BC3 decode and re-encode
 *   BC_OUT_ASTC  -- BC1/BC2/BC3 -> ASTC 4x4 repack, BC7 -> ASTC 4x4 decode and re-encode (PanVK)
 *   BC_OUT_ASTC7 -- BC7 -> ASTC 4x4 decode and re-encode with an endpoint and mode search (RADV) */
layout(set = 0, binding = 1, rgba32ui) uniform writeonly BC_UIMAGE o_img;
#elif defined(BC_OUT_EAC_R)
/* BC4 -> EAC_R11: an 8-byte destination block, one rg32ui texel. */
layout(set = 0, binding = 1, rg32ui) uniform writeonly BC_UIMAGE o_img;
#elif defined(BC_OUT_RG11B10)
/* A B10G11R11_UFLOAT plane, packed here (R in bits 0-10, G 11-21, B 22-31). */
layout(set = 0, binding = 1, r32ui) uniform writeonly BC_UIMAGE o_img;
#elif defined(BC_OUT_SNORM8)
layout(set = 0, binding = 1, rgba8_snorm) uniform writeonly BC_IMAGE o_img;
#elif defined(BC_OUT_F16)
layout(set = 0, binding = 1, rgba16f) uniform writeonly BC_IMAGE o_img;
#else
layout(set = 0, binding = 1, rgba8) uniform writeonly BC_IMAGE o_img;
#endif

/* Endpoint-spread histogram: hardware BC4 is exact when spread <= 32. Samples 1 in 64 blocks. */
layout(set = 0, binding = 2, std430) buffer Hist { uint bucket[]; } hist;
/* The histogram is the first 1024 dwords; the [BCCONTENT] ring follows in the same buffer. */
#define BC_HASH_BASE 1024u

#if defined(BC_OUT_EAC) || defined(BC_OUT_EAC_R)
#if defined(BC_TABLES_BUFFER)
/* Per BC4 endpoint pair, the best EAC_R11 config: base<<8 | mult<<4 | table. Precomputed offline
 * (eac_lut.h), exact for spread <= 32. Two 16-bit entries per uint. */
layout(set = 0, binding = 3, std430) readonly buffer EacLut { uint pair[]; } eaclut;
#else
/* RADV: the same tables interleaved, one 16-byte entry per BC4 endpoint pair (radv_meta_bc_decode.c):
 * x = both configs, y / z = the two index remaps. A channel reads one entry, one cache line. */
#define BC_EAC_PACKED
layout(set = 0, binding = 3, std430) readonly buffer EacLut { uvec4 ent[]; } eaclut;
#endif
#endif

#if defined(BC_TABLES_BUFFER)
/* PanVK: the lookup tables below come from this buffer instead of constant arrays. Bifrost
 * has no constant-data path, so every constant array indexed with a variable was copied to
 * the stack of every thread, 10 KB per thread for BC7 -> ASTC. panvk_bc_tables_gen.py
 * builds the buffer from the constant definitions, in this member order. */
layout(set = 0, binding = 4, std430) readonly buffer BcTables {
   uint bc_part2[64];
   uint bc_part3[64];
   int bc_anchor2[64];
   int bc_anchor3b[64];
   int bc_anchor3c[64];
   int bc6_flags[18];
   int bc6_npb[18];
   int bc6_neb[18];
   int bc6_nib[18];
   int bc6_d0[18];
   int bc6_d1[18];
   int bc6_d2[18];
   int bc6_foff[19];
   uint bc6_fields[236];
   int bc_weights[28];
   int bc7_nsub[8];
   int bc7_npb[8];
   int bc7_rot[8];
   int bc7_isel[8];
   int bc7_cbits[8];
   int bc7_abits[8];
   int bc7_epb[8];
   int bc7_spb[8];
   int bc7_ib[8];
   int bc7_ib2[8];
   uint astc_trit_enc[61];
   uint astc_q48_unq[12];
   uint astc_q48_near[64];
   uint astc_q128_near[64];
};
#endif

layout(push_constant) uniform push_constants {
   ivec3 offset;
   int format;
   int image_type;
   /* Two scalars, not ivec2: ivec2 alignment would pad to offset 24 and break the packed layout. */
   int extent_x;
   int extent_y;
   /* Ring slot for the [BCCONTENT] census; -1 disables. */
   int hash_slot;
   /* BC_SRC_BUFFER only: the source blocks, row and slice pitch in blocks. */
   uvec2 src_va;
   uint src_row_blocks;
   uint src_slice_blocks;
} pc;

/* Keep in sync with enum vk_texcompress_bc_format. */
#define BC_FMT_BC4_UNORM  0
#define BC_FMT_BC4_SNORM  1
#define BC_FMT_BC5_UNORM  2
#define BC_FMT_BC5_SNORM  3
#define BC_FMT_BC6H_UF16  4
#define BC_FMT_BC6H_SF16  5
#define BC_FMT_BC7        6
/* Only the ASTC variant sees these: BC1-BC3 are transcoded only where nothing decodes them. */
#define BC_FMT_BC1_RGB    7
#define BC_FMT_BC1_RGBA   8
#define BC_FMT_BC2        9
#define BC_FMT_BC3        10

#if !defined(BC_TABLES_BUFFER)
const uint bc_part2[64] = uint[](0x50505050u, 0x40404040u, 0x54545454u, 0x54505040u, 0x50404000u, 0x55545450u, 0x55545040u, 0x54504000u, 0x50400000u, 0x55555450u, 0x55544000u, 0x54400000u, 0x55555440u, 0x55550000u, 0x55555500u, 0x55000000u, 0x55150100u, 0x00004054u, 0x15010000u, 0x00405054u, 0x00004050u, 0x15050100u, 0x05010000u, 0x40505054u, 0x00404050u, 0x05010100u, 0x14141414u, 0x05141450u, 0x01155440u, 0x00555500u, 0x15014054u, 0x05414150u, 0x44444444u, 0x55005500u, 0x11441144u, 0x05055050u, 0x05500550u, 0x11114444u, 0x41144114u, 0x44111144u, 0x15055054u, 0x01055040u, 0x05041050u, 0x05455150u, 0x14414114u, 0x50050550u, 0x41411414u, 0x00141400u, 0x00041504u, 0x00105410u, 0x10541000u, 0x04150400u, 0x50410514u, 0x41051450u, 0x05415014u, 0x14054150u, 0x41050514u, 0x41505014u, 0x40011554u, 0x54150140u, 0x50505500u, 0x00555050u, 0x15151010u, 0x54540404u);
const uint bc_part3[64] = uint[](0xaa685050u, 0x6a5a5040u, 0x5a5a4200u, 0x5450a0a8u, 0xa5a50000u, 0xa0a05050u, 0x5555a0a0u, 0x5a5a5050u, 0xaa550000u, 0xaa555500u, 0xaaaa5500u, 0x90909090u, 0x94949494u, 0xa4a4a4a4u, 0xa9a59450u, 0x2a0a4250u, 0xa5945040u, 0x0a425054u, 0xa5a5a500u, 0x55a0a0a0u, 0xa8a85454u, 0x6a6a4040u, 0xa4a45000u, 0x1a1a0500u, 0x0050a4a4u, 0xaaa59090u, 0x14696914u, 0x69691400u, 0xa08585a0u, 0xaa821414u, 0x50a4a450u, 0x6a5a0200u, 0xa9a58000u, 0x5090a0a8u, 0xa8a09050u, 0x24242424u, 0x00aa5500u, 0x24924924u, 0x24499224u, 0x50a50a50u, 0x500aa550u, 0xaaaa4444u, 0x66660000u, 0xa5a0a5a0u, 0x50a050a0u, 0x69286928u, 0x44aaaa44u, 0x66666600u, 0xaa444444u, 0x54a854a8u, 0x95809580u, 0x96969600u, 0xa85454a8u, 0x80959580u, 0xaa141414u, 0x96960000u, 0xaaaa1414u, 0xa05050a0u, 0xa0a5a5a0u, 0x96000000u, 0x40804080u, 0xa9a8a9a8u, 0xaaaaaa44u, 0x2a4a5254u);
const int bc_anchor2[64] = int[](15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 2, 8, 2, 2, 8, 8, 15, 2, 8, 2, 2, 8, 8, 2, 2, 15, 15, 6, 8, 2, 8, 15, 15, 2, 8, 2, 2, 2, 15, 15, 6, 6, 2, 6, 8, 15, 15, 2, 2, 15, 15, 15, 15, 15, 2, 2, 15);
const int bc_anchor3b[64] = int[](3, 3, 15, 15, 8, 3, 15, 15, 8, 8, 6, 6, 6, 5, 3, 3, 3, 3, 8, 15, 3, 3, 6, 10, 5, 8, 8, 6, 8, 5, 15, 15, 8, 15, 3, 5, 6, 10, 8, 15, 15, 3, 15, 5, 15, 15, 15, 15, 3, 15, 5, 5, 5, 8, 5, 10, 5, 10, 8, 13, 15, 12, 3, 3);
const int bc_anchor3c[64] = int[](15, 8, 8, 3, 15, 15, 3, 8, 15, 15, 15, 15, 15, 15, 15, 8, 15, 8, 15, 3, 15, 8, 15, 8, 3, 15, 6, 10, 15, 15, 10, 8, 15, 3, 15, 10, 10, 8, 9, 10, 6, 15, 8, 15, 3, 6, 6, 8, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 3, 15, 15, 8);
#endif

/* BC6H mode tables, generated from Mesa texcompress_bptc_tmp.h. Do not hand-edit. */
#if !defined(BC_TABLES_BUFFER)
const int bc6_flags[18] = int[](2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 1, 2, 1, 2, 1, 0, 1);
const int bc6_npb[18] = int[](5, 5, 5, 0, 5, 0, 5, 0, 5, 0, 5, 0, 5, 0, 5, 0, 5, 0);
const int bc6_neb[18] = int[](10, 7, 11, 10, 11, 11, 11, 12, 9, 16, 8, 0, 8, 0, 8, 0, 6, 0);
const int bc6_nib[18] = int[](3, 3, 3, 4, 3, 4, 3, 4, 3, 4, 3, 0, 3, 0, 3, 0, 3, 0);
const int bc6_d0[18] = int[](5, 6, 5, 10, 4, 9, 4, 8, 5, 4, 6, 0, 5, 0, 5, 0, 6, 0);
const int bc6_d1[18] = int[](5, 6, 4, 10, 5, 9, 4, 8, 5, 4, 5, 0, 6, 0, 5, 0, 6, 0);
const int bc6_d2[18] = int[](5, 6, 4, 10, 4, 9, 5, 8, 5, 4, 5, 0, 5, 0, 6, 0, 6, 0);
const int bc6_foff[19] = int[](0, 19, 42, 60, 66, 86, 95, 115, 124, 143, 152, 171, 171, 192, 192, 213, 213, 236, 236);
const uint bc6_fields[236] = uint[](326u, 330u, 331u, 2560u, 2564u, 2568u, 1281u, 327u, 1030u, 1285u, 267u, 1031u, 1289u, 283u, 1034u, 1282u, 299u, 1283u, 315u, 342u, 327u, 343u, 1792u, 267u, 283u, 330u, 1796u, 346u, 299u, 326u, 1800u, 315u, 347u, 331u, 1537u, 1030u, 1541u, 1031u, 1545u, 1034u, 1538u, 1539u, 2560u, 2564u, 2568u, 1281u, 416u, 1030u, 1029u, 420u, 267u, 1031u, 1033u, 424u, 283u, 1034u, 1282u, 299u, 1283u, 315u, 2560u, 2564u, 2568u, 2561u, 2565u, 2569u, 2560u, 2564u, 2568u, 1025u, 416u, 327u, 1030u, 1285u, 420u, 1031u, 1033u, 424u, 283u, 1034u, 1026u, 267u, 299u, 1027u, 326u, 315u, 2560u, 2564u, 2568u, 2305u, 416u, 2309u, 420u, 2313u, 424u, 2560u, 2564u, 2568u, 1025u, 416u, 330u, 1030u, 1029u, 420u, 267u, 1031u, 1289u, 424u, 1034u, 1026u, 283u, 299u, 1027u, 331u, 315u, 2560u, 2564u, 2568u, 2049u, 8864u, 2053u, 8868u, 2057u, 8872u, 2304u, 330u, 2308u, 326u, 2312u, 331u, 1281u, 327u, 1030u, 1285u, 267u, 1031u, 1289u, 283u, 1034u, 1282u, 299u, 1283u, 315u, 2560u, 2564u, 2568u, 1025u, 9888u, 1029u, 9892u, 1033u, 9896u, 2048u, 327u, 330u, 2052u, 299u, 326u, 2056u, 315u, 331u, 1537u, 1030u, 1285u, 267u, 1031u, 1289u, 283u, 1034u, 1538u, 1539u, 2048u, 267u, 330u, 2052u, 342u, 326u, 2056u, 343u, 331u, 1281u, 327u, 1030u, 1541u, 1031u, 1289u, 283u, 1034u, 1282u, 299u, 1283u, 315u, 2048u, 283u, 330u, 2052u, 346u, 326u, 2056u, 347u, 331u, 1281u, 327u, 1030u, 1285u, 267u, 1031u, 1545u, 1034u, 1282u, 299u, 1283u, 315u, 1536u, 327u, 267u, 283u, 330u, 1540u, 342u, 346u, 299u, 326u, 1544u, 343u, 315u, 347u, 331u, 1537u, 1030u, 1541u, 1031u, 1545u, 1034u, 1538u, 1539u);
#endif

/* ---------------------------------------------------------------- bit access */

/* Extract n bits at offset `off` from a 128-bit block in constant time (a field spans <= 2 dwords).
 * Guards: n <= 0 (bc7_npb[] is 0 for modes 4-6, 1u << 32 is undefined), s != 0 (shift by 32),
 * w < 3 (blk[4] out of bounds). */
uint bc_bits(uvec4 blk, int off, int n)
{
   if (n <= 0)
      return 0u;

   int w = off >> 5;
   int s = off & 31;

   uint v = blk[w] >> uint(s);
   if (s != 0 && w < 3)
      v |= blk[w + 1] << uint(32 - s);

   return n >= 32 ? v : (v & ((1u << uint(n)) - 1u));
}

/* 32 bits of the block from bit `off` on; bits past 127 read as 0. */
uint bc_win32(uvec4 blk, int off)
{
   const int w = off >> 5, s = off & 31;
   const uint a = w == 0 ? blk.x : (w == 1 ? blk.y : (w == 2 ? blk.z : (w == 3 ? blk.w : 0u)));
   const uint b = w == 0 ? blk.y : (w == 1 ? blk.z : (w == 2 ? blk.w : 0u));
   return s == 0 ? a : ((a >> uint(s)) | (b << uint(32 - s)));
}

/* A 64-bit index stream shifted right by n (0 <= n < 32). */
uvec2 bc_shr64(uvec2 st, int n)
{
   return uvec2(n == 0 ? st.x : ((st.x >> uint(n)) | (st.y << uint(32 - n))), st.y >> uint(n));
}

/* Buckets 0..5 = spread <=8, 9-16, 17-32, 33-64, 65-128, >128; 6 = total sampled;
 * 7 = r0 > r1 (8-value) mode; 8 = 8-value mode and spread <= 32. */
void bc_record_spread(uvec4 blk, int base_bit, int r0, int r1)
{
   int s = r0 > r1 ? r0 - r1 : r1 - r0;
   int b = s <= 8 ? 0 : s <= 16 ? 1 : s <= 32 ? 2 : s <= 64 ? 3 : s <= 128 ? 4 : 5;
   atomicAdd(hist.bucket[b], 1u);
   atomicAdd(hist.bucket[6], 1u);
   if (r0 > r1)
      atomicAdd(hist.bucket[7], 1u);

   /* Representable = every texel used lies on the endpoint line. The 6-value mode is only off the
    * line when indices 6 or 7 (0.0 / 1.0) are used. */
   bool offline = false;
   if (r0 <= r1) {
      for (int t = 0; t < 16; t++) {
         uint idx = bc_bits(blk, base_bit + 16 + 3 * t, 3);
         if (idx >= 6u) { offline = true; break; }
      }
   }
   if (!offline && s <= 32)
      atomicAdd(hist.bucket[8], 1u);
   if (offline)
      atomicAdd(hist.bucket[9], 1u);
}

/* BC5 census: do R and G span the same range? ASTC 4x4 dual-plane with CEM 0 could carry BC5, but
 * both planes share one [L0, L1]. Reports the distribution, not a mean. */
void bc_record_rg(int r0, int r1, int g0, int g1)
{
   /* Compare spans, not raw endpoints: BC4 stores them in either order. */
   int rlo = min(r0, r1), rhi = max(r0, r1);
   int glo = min(g0, g1), ghi = max(g0, g1);
   int dlo = abs(rlo - glo), dhi = abs(rhi - ghi);
   int d = max(dlo, dhi);

   int b = d == 0 ? 0 : d <= 2 ? 1 : d <= 4 ? 2 : d <= 8 ? 3 : d <= 16 ? 4 : d <= 32 ? 5 : 6;
   atomicAdd(hist.bucket[10 + b], 1u);
   atomicAdd(hist.bucket[17], 1u);
}


#if defined(BC_OUT_EAC) || defined(BC_OUT_EAC_R)
/* ---- BC5 -> EAC_R11G11 encoder ----
 * Both are 16 bytes, two channels, 3-bit per-texel indices, so only the palette is re-encoded.
 * Tables, value formula, field positions and texel order verified on hardware. */
const int bc_eac_mod[128] = int[128](
   -3, -6,  -9, -15, 2, 5, 8, 14,   -3, -7, -10, -13, 2, 6, 9, 12,
   -2, -5,  -8, -13, 1, 4, 7, 12,   -2, -4,  -6, -13, 1, 3, 5, 12,
   -3, -6,  -8, -12, 2, 5, 7, 11,   -3, -7,  -9, -11, 2, 6, 8, 10,
   -4, -7,  -8, -11, 3, 6, 7, 10,   -3, -5,  -8, -11, 2, 4, 7, 10,
   -2, -6,  -8, -10, 1, 5, 7,  9,   -2, -5,  -8, -10, 1, 4, 7,  9,
   -2, -4,  -8, -10, 1, 3, 7,  9,   -2, -5,  -7, -10, 1, 4, 6,  9,
   -3, -4,  -7, -10, 2, 3, 6,  9,   -1, -2,  -3, -10, 0, 1, 2,  9,
   -4, -6,  -8,  -9, 3, 5, 7,  8,   -3, -5,  -7,  -9, 2, 4, 6,  8);

/* One BC4 palette entry, per the format definition. */
int bc_eac_bc4val(int i, int e0, int e1)
{
   if (i == 0) return e0;
   if (i == 1) return e1;
   if (e0 > e1) return ((8 - i) * e0 + (i - 1) * e1) / 7;
   if (i == 6) return 0;
   if (i == 7) return 255;
   return ((6 - i) * e0 + (i - 1) * e1) / 5;
}

/* Place a 3-bit index at bit p of the 64-bit EAC word (hiW = bits 63..32, loW = 31..0).
 * p = 30 and 31 straddle both halves. */
void bc_eac_put3(inout uint hiW, inout uint loW, int p, uint v)
{
   if (p >= 32)      hiW |= v << uint(p - 32);
   else if (p <= 29) loW |= v << uint(p);
   else            { loW |= v << uint(p); hiW |= v >> uint(32 - p); }
}

/* Encode one 8-byte BC4 channel block into one 8-byte EAC_R11 block.
 * `half01` is the channel's two source uints; returns the EAC block in MEMORY order.
 *
 * snorm (BC4/BC5_SNORM -> EAC_R11/R11G11_SNORM): the UNORM fit runs on the endpoints moved to offset
 * binary (e + 128 keeps their order, the palette mode and its shape), then the base codeword moves
 * back by 128. SNORM decodes base*8 + d where UNORM decodes base*8 + 4 + d, so the result is the
 * UNORM fit shifted by 1028: within 4/1023 of the SNORM target. -128 is not a SNORM base: -127. */
#if defined(BC_EAC_PACKED)
/* The 16 BC4 indices (3 bits each, texel t at bit 3t) in two 24-bit halves. */
uint bc_eac_idx(uint a, uint b, int t)
{
   return ((t < 8 ? a : b) >> uint(3 * (t & 7))) & 7u;
}

uvec2 bc_eac_encode(uvec2 half01, bool snorm)
{
   if (snorm)
      half01.x ^= 0x8080u;
   const uint e0 = half01.x & 0xffu;
   const uint e1 = (half01.x >> 8) & 0xffu;
   const uint a = (half01.x >> 16) | ((half01.y & 0xffu) << 16);
   const uint b = half01.y >> 8;

   /* In the 6-value mode (e0 <= e1) indices 6/7 (0.0, 1.0) are off the line; if the block uses
    * them, take the config fitted to all eight entries. An index is 6 or 7 when its bits 2 and 1
    * are both set, tested for all sixteen at once. */
   const uint m = 0x492492u;
   const bool uses_offline = e0 <= e1 && (((a & (a >> 1)) | (b & (b >> 1))) & m) != 0u;
   const uvec4 ent = eaclut.ent[e0 * 256u + e1];
   const uint cfg = uses_offline ? (ent.x >> 16u) : (ent.x & 0xffffu);
   const uint rw = uses_offline ? ent.z : ent.y;
   const int base_cw = snorm ? (max(int(cfg >> 8u) - 128, -127) & 0xff) : int(cfg >> 8u);

   /* Texel order: EAC slot j is texel (j/4, j%4), column-major; BC4 texel t is (t%4, t/4).
    * So texel t goes to slot (t%4)*4 + t/4, its 3 bits at 45 - 3*slot of the 64-bit word. The loop
    * unrolls, so every position is a constant. */
   uint hiW = (uint(base_cw) << 24) | ((cfg & 0xffu) << 16);
   uint loW = 0u;
   for (int t = 0; t < 16; t++) {
      const uint v = (rw >> (3u * bc_eac_idx(a, b, t))) & 7u;
      const int p = 45 - 3 * ((t % 4) * 4 + (t / 4));
      if (p >= 32)
         hiW |= v << uint(p - 32);
      else if (p <= 29)
         loW |= v << uint(p);
      else {
         loW |= v << uint(p);
         hiW |= v >> uint(32 - p);
      }
   }

   /* The 64-bit word is big-endian and the image is read as little-endian uints: byte-reverse. */
   return uvec2(((hiW & 0xffu) << 24) | ((hiW & 0xff00u) << 8) | ((hiW >> 8) & 0xff00u) | (hiW >> 24),
                ((loW & 0xffu) << 24) | ((loW & 0xff00u) << 8) | ((loW >> 8) & 0xff00u) | (loW >> 24));
}
#else
uvec2 bc_eac_encode(uvec2 half01, bool snorm)
{
   if (snorm)
      half01.x ^= 0x8080u;
   const int e0 = int(half01.x & 0xffu);
   const int e1 = int((half01.x >> 8) & 0xffu);

   /* Two configs per endpoint pair. In the 6-value mode indices 6/7 (0.0, 1.0) are off the line;
    * if the block uses them, take the config fitted to all eight entries. */
   bool uses_offline = false;
   if (e0 <= e1) {
      for (int t = 0; t < 16; t++)
         if (bc_bits(uvec4(half01, 0u, 0u), 16 + 3 * t, 3) >= 6u) { uses_offline = true; break; }
   }
   const uint li = uint(e0 * 256 + e1);
   const uint word = eaclut.pair[li];
   const uint cfg = uses_offline ? (word >> 16u) : (word & 0xffffu);
   const int base_cw = snorm ? (max(int(cfg >> 8u) - 128, -127) & 0xff) : int(cfg >> 8u);
   const int mult    = int((cfg >> 4u) & 0xfu);
   const int tab     = int(cfg & 0xfu);

   /* Nearest EAC index for each of the 8 BC4 entries, looked up by (e0, e1, uses_offline).
    * Packed 3 bits per entry to avoid a dynamically indexed local array. */
   const uint rw = eaclut.pair[65536u + li * 2u + (uses_offline ? 1u : 0u)];

   /* Texel order: EAC slot j is texel (j/4, j%4), column-major; BC4 texel t is (t%4, t/4).
    * So texel t goes to slot (t%4)*4 + t/4. */
   uint hiW = (uint(base_cw) << 24) | (uint(mult) << 20) | (uint(tab) << 16);
   uint loW = 0u;
   for (int t = 0; t < 16; t++) {
      const uint bi = bc_bits(uvec4(half01, 0u, 0u), 16 + 3 * t, 3);
      const int j = (t % 4) * 4 + (t / 4);
      bc_eac_put3(hiW, loW, 45 - 3 * j, (rw >> (3u * bi)) & 7u);
   }

   /* The 64-bit word is big-endian and the image is read as little-endian uints: byte-reverse. */
   return uvec2(((hiW & 0xffu) << 24) | ((hiW & 0xff00u) << 8) | ((hiW >> 8) & 0xff00u) | (hiW >> 24),
                ((loW & 0xffu) << 24) | ((loW & 0xff00u) << 8) | ((loW >> 8) & 0xff00u) | (loW >> 24));
}
#endif
#endif

int bc_sign_extend(int v, int bits)
{
   int m = 1 << (bits - 1);
   return (v ^ m) - m;
}

/* The three weight tables in one array (2-bit at 0, 3-bit at 4, 4-bit at 12): one dynamic index
 * instead of three. */
#if !defined(BC_TABLES_BUFFER)
const int bc_weights[28] = int[](
   0, 21, 43, 64,                                                          /* 2-bit */
   0, 9, 18, 27, 37, 46, 55, 64,                                           /* 3-bit */
   0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64);           /* 4-bit */
#endif

int bc_interpolate(int a, int b, int index, int index_bits)
{
   int w = bc_weights[(index_bits == 2 ? 0 : (index_bits == 3 ? 4 : 12)) + index];
   return (a * (64 - w) + b * w + 32) >> 6;
}

#if defined(BC_OUT_UNORM8) || defined(BC_OUT_BC3) || defined(BC_OUT_ASTC) || defined(BC_OUT_ASTC7)
/* The decoded BC7 block, RGBA8 per texel, four texels per vector. Mali keeps any array of more
 * than 8 entries on the stack of every thread (768 threads per core on the G76), so no 16-texel
 * array: bc7_decode_block() fills this, the encoders read it back. */
uvec4 bc_px[4];

ivec4 bc_px_get(int t)
{
   const uint p = bc_px[t >> 2][t & 3];
   return ivec4(int(p & 0xffu), int((p >> 8) & 0xffu), int((p >> 16) & 0xffu), int(p >> 24));
}

/* The same texel as floats. Converting the unsigned byte is one v_cvt_f32_ubyteN; converting the
 * int from bc_px_get() is an extract and a v_cvt_f32_i32. */
vec4 bc_px_f(int t)
{
   const uint p = bc_px[t >> 2][t & 3];
   return vec4(float(p & 0xffu), float((p >> 8) & 0xffu), float((p >> 16) & 0xffu), float(p >> 24));
}
#endif

#if defined(BC_OUT_BC3)
/* ---- BC7 -> BC3 (DXT5) encoder ----
 * No table search: both halves are a line between the block's extremes and indices are a
 * projection, O(1) per texel. Costs 2-bit colour indices and RGB565 endpoints (gradient banding).
 * BC1 mode depends on endpoint order: c0 > c1 is 4-colour opaque, c0 <= c1 makes index 3
 * transparent. Endpoint order is handled explicitly below. */

/* Level along the line (0 = c1, 3 = c0) -> BC1 index. BC1 palette order is c0, c1, interpolants. */
const int bc_dxt_cidx[4] = int[](1, 3, 2, 0);

void bc_dxt_put(inout uint hiW, inout uint loW, int off, int n, uint v)
{
   for (int i = 0; i < n; i++)
      if ((v & (1u << uint(i))) != 0u) {
         int b = off + i;
         if (b >= 32) hiW |= 1u << uint(b - 32);
         else         loW |= 1u << uint(b);
      }
}

uvec4 bc_bc3_encode()
{
   /* ---- alpha half: a BC4 block. Endpoints are the extremes; 8 levels, 3-bit indices. ---- */
   int amin = 255, amax = 0;
   for (int t = 0; t < 16; t++) {
      amin = min(amin, bc_px_get(t).a);
      amax = max(amax, bc_px_get(t).a);
   }
   uint ahi = 0u, alo = 0u;
   bc_dxt_put(ahi, alo, 0, 8, uint(amax));   /* alpha0 > alpha1 selects the 8-value mode */
   bc_dxt_put(ahi, alo, 8, 8, uint(amin));
   if (amax > amin) {
      const int rng = amax - amin;
      for (int t = 0; t < 16; t++) {
         /* Index 0 = a0, 1 = a1, 2..7 interpolate down from a0, so level q maps to 8-q.
          * q = (num / rng) for num < 8 * rng, counted with compares instead of a division. */
         const int num = (bc_px_get(t).a - amin) * 7 + rng / 2;
         int q = 0, thr = rng;
         for (int k = 1; k < 8; k++, thr += rng)
            q += num >= thr ? 1 : 0;
         const int idx = (q == 7) ? 0 : ((q == 0) ? 1 : (8 - q));
         bc_dxt_put(ahi, alo, 16 + 3 * t, 3, uint(idx));
      }
   }
   /* amax == amin leaves every index 0, which decodes to alpha0 in either mode. */

   /* ---- colour half: a BC1 block. ---- */
   ivec3 lo = ivec3(255), hi = ivec3(0);
   for (int t = 0; t < 16; t++) {
      lo = min(lo, bc_px_get(t).rgb);
      hi = max(hi, bc_px_get(t).rgb);
   }
   /* Inset the bounding box by 1/16 of its extent (standard fast-DXT refinement). */
   const ivec3 ins = (hi - lo) / 16;
   vec3 f0 = vec3(hi - ins);
   vec3 f1 = vec3(lo + ins);

   /* Refine the box diagonal, which misses two-cluster blocks (BC7 mode 7): assign indices, solve
    * least-squares endpoints per channel with weights u in {0, 1/3, 2/3, 1}, then re-assign:
    *      [ S(1-u)^2   S(1-u)u ] [E1]   [ S(1-u)c ]
    *      [ S(1-u)u    S u^2   ] [E0] = [ S u c   ]
    * Two iterations (85.9% -> 90.2% -> 90.5% of texels within 10 LSB). */
   for (int pass = 0; pass < 2; pass++) {
      const ivec3 pq0 = ivec3(int(f0.r) >> 3, int(f0.g) >> 2, int(f0.b) >> 3);
      const ivec3 pq1 = ivec3(int(f1.r) >> 3, int(f1.g) >> 2, int(f1.b) >> 3);
      const ivec3 pd0 = ivec3((pq0.r << 3) | (pq0.r >> 2), (pq0.g << 2) | (pq0.g >> 4), (pq0.b << 3) | (pq0.b >> 2));
      const ivec3 pd1 = ivec3((pq1.r << 3) | (pq1.r >> 2), (pq1.g << 2) | (pq1.g >> 4), (pq1.b << 3) | (pq1.b >> 2));
      const ivec3 pdir = pd0 - pd1;
      const int pden = pdir.r * pdir.r + pdir.g * pdir.g + pdir.b * pdir.b;
      if (pden <= 0)
         break;

      /* The projection in float is exact (every product and sum is an integer below 2^24), and
       * the index (num * 3 + pden / 2) / pden of a clamped num is below 4: three compares
       * instead of a 32-bit multiply per channel and an integer division per texel. */
      const vec3 pdirf = vec3(pdir), pd1f = vec3(pd1);
      const float pdenf = float(pden), phalf = float(pden / 2);
      float A = 0.0, B = 0.0, C = 0.0;
      vec3 P = vec3(0.0), Q = vec3(0.0);
      for (int t = 0; t < 16; t++) {
         const vec3 wt = bc_px_f(t).rgb;
         const float num = clamp(dot(wt - pd1f, pdirf), 0.0, pdenf) * 3.0 + phalf;
         const float u = float((num >= pdenf ? 1 : 0) + (num >= 2.0 * pdenf ? 1 : 0) +
                               (num >= 3.0 * pdenf ? 1 : 0)) / 3.0;
         const float v = 1.0 - u;
         A += v * v; B += v * u; C += u * u;
         P += v * wt; Q += u * wt;
      }
      const float det = A * C - B * B;
      if (abs(det) < 1e-4)
         break;                         /* every texel took the same index: nothing to solve */
      f1 = clamp((C * P - B * Q) / det, vec3(0.0), vec3(255.0));
      f0 = clamp((A * Q - B * P) / det, vec3(0.0), vec3(255.0));
   }

   ivec3 q0 = ivec3(int(f0.r) >> 3, int(f0.g) >> 2, int(f0.b) >> 3);
   ivec3 q1 = ivec3(int(f1.r) >> 3, int(f1.g) >> 2, int(f1.b) >> 3);
   uint p0 = uint((q0.r << 11) | (q0.g << 5) | q0.b);
   uint p1 = uint((q1.r << 11) | (q1.g << 5) | q1.b);
   if (p0 < p1) {                       /* keep c0 > c1: see the punch-through trap above */
      const uint pt = p0; p0 = p1; p1 = pt;
      const ivec3 qt = q0; q0 = q1; q1 = qt;
   }

   /* Dequantise exactly as the texture unit will, replicating high bits into the low ones. */
   const ivec3 d0 = ivec3((q0.r << 3) | (q0.r >> 2), (q0.g << 2) | (q0.g >> 4), (q0.b << 3) | (q0.b >> 2));
   const ivec3 d1 = ivec3((q1.r << 3) | (q1.r >> 2), (q1.g << 2) | (q1.g >> 4), (q1.b << 3) | (q1.b >> 2));

   const ivec3 dir = d0 - d1;
   const int den = dir.r * dir.r + dir.g * dir.g + dir.b * dir.b;
   uint cidx = 0u;
   if (den > 0) {
      const vec3 dirf = vec3(dir), d1f = vec3(d1);
      const float denf = float(den), half_den = float(den / 2);
      for (int t = 0; t < 16; t++) {
         const float num = clamp(dot(bc_px_f(t).rgb - d1f, dirf), 0.0, denf) * 3.0 + half_den;
         const int q = (num >= denf ? 1 : 0) + (num >= 2.0 * denf ? 1 : 0) + (num >= 3.0 * denf ? 1 : 0);
         cidx |= uint(bc_dxt_cidx[q]) << uint(2 * t);
      }
   }
   /* den == 0: endpoints quantised equal; index 0 is still that colour in 3-colour mode. */

   /* BC3 is little-endian throughout -- no byte reversal, unlike the EAC carrier. */
   return uvec4(alo, ahi, p0 | (p1 << 16), cidx);
}
#endif

bool bc_is_anchor(int n_subsets, int part, int texel)
{
   if (texel == 0)
      return true;
   if (n_subsets == 2)
      return bc_anchor2[part] == texel;
   if (n_subsets == 3)
      return bc_anchor3b[part] == texel || bc_anchor3c[part] == texel;
   return false;
}

int bc_anchors_before(int n_subsets, int part, int texel)
{
   if (texel == 0)
      return 0;
   int count = 1;
   if (n_subsets == 2) {
      if (texel > bc_anchor2[part])
         count++;
   } else if (n_subsets == 3) {
      if (texel > bc_anchor3b[part])
         count++;
      if (texel > bc_anchor3c[part])
         count++;
   }
   return count;
}

/* ------------------------------------------------------------ BC4 / BC5 (RGTC) */

/* One RGTC channel of a whole 4x4 block. The palette is built once; only the index is per texel.
 * Palette expressions keep the original order, so float results are bit-exact. */
void bc_rgtc_block(uvec4 blk, int base_bit, bool is_signed, out float out_vals[16])
{
   uint r0 = bc_bits(blk, base_bit, 8);
   uint r1 = bc_bits(blk, base_bit + 8, 8);

   float e0, e1;
   bool eight;
   if (is_signed) {
      int s0 = bc_sign_extend(int(r0), 8);
      int s1 = bc_sign_extend(int(r1), 8);
      if (s0 == -128) s0 = -127;
      if (s1 == -128) s1 = -127;
      e0 = float(s0) / 127.0;
      e1 = float(s1) / 127.0;
      eight = s0 > s1;
   } else {
      e0 = float(r0) / 255.0;
      e1 = float(r1) / 255.0;
      eight = r0 > r1;
   }

   float pal[8];
   pal[0] = e0;
   pal[1] = e1;
   if (eight) {
      for (int i = 2; i < 8; i++)
         pal[i] = mix(e0, e1, float(uint(i) - 1u) / 7.0);
   } else {
      for (int i = 2; i < 6; i++)
         pal[i] = mix(e0, e1, float(uint(i) - 1u) / 5.0);
      pal[6] = is_signed ? -1.0 : 0.0;
      pal[7] = 1.0;
   }

   /* Per texel: one bit extract and one palette read. base_bit is not mutated. */
   for (int texel = 0; texel < 16; texel++)
      out_vals[texel] = pal[bc_bits(blk, base_bit + 16 + 3 * texel, 3)];
}

/* ------------------------------------------------------------------- BC7 */

/* Per-mode parameters (Mesa's bptc_unorm_modes). */
#if !defined(BC_TABLES_BUFFER)
const int bc7_nsub[8]  = int[](3, 2, 3, 2, 1, 1, 1, 2);
const int bc7_npb[8]   = int[](4, 6, 6, 6, 0, 0, 0, 6);
const int bc7_rot[8]   = int[](0, 0, 0, 0, 1, 1, 0, 0);
const int bc7_isel[8]  = int[](0, 0, 0, 0, 1, 0, 0, 0);
const int bc7_cbits[8] = int[](4, 6, 5, 7, 5, 7, 7, 5);
const int bc7_abits[8] = int[](0, 0, 0, 0, 6, 8, 7, 5);
const int bc7_epb[8]   = int[](1, 0, 0, 1, 0, 0, 1, 1);
const int bc7_spb[8]   = int[](0, 1, 0, 0, 0, 0, 0, 0);
const int bc7_ib[8]    = int[](3, 3, 2, 2, 2, 2, 4, 2);
const int bc7_ib2[8]   = int[](0, 0, 0, 0, 3, 2, 0, 0);
#endif

/* BC7 census: which modes and partitions does content use? BC7 weights equal ASTC's, but only
 * 30/64 two-subset and 11/64 three-subset partitions are reproducible in ASTC.
 * Modes 4/5/6 are single-subset. */
const uint bc7_ok2_lo = 0x27feffffu, bc7_ok2_hi = 0x00100003u;   /* the 30 reproducible 2-subset */
const uint bc7_ok3_lo = 0x00103f10u, bc7_ok3_hi = 0x02000018u;   /* the 11 reproducible 3-subset */

void bc_record_bc7(uvec4 blk)
{
   /* Mode is a unary prefix: the position of the lowest set bit. */
   int mode = -1;
   for (int m = 0; m < 8; m++)
      if ((blk.x & (1u << uint(m))) != 0u) { mode = m; break; }
   if (mode < 0)
      return;                                   /* reserved encoding */

   atomicAdd(hist.bucket[18 + mode], 1u);
   atomicAdd(hist.bucket[30], 1u);

   const int nsub = bc7_nsub[mode];
   if (nsub == 1)
      return;                                   /* modes 4/5/6 -- no partition to match */

   const int part = int(bc_bits(blk, mode + 1, bc7_npb[mode]));
   const uint lo = nsub == 2 ? bc7_ok2_lo : bc7_ok3_lo;
   const uint hi = nsub == 2 ? bc7_ok2_hi : bc7_ok3_hi;
   const bool ok = part < 32 ? (lo & (1u << uint(part))) != 0u
                             : (hi & (1u << uint(part - 32))) != 0u;

   atomicAdd(hist.bucket[(nsub == 2 ? 26 : 28) + (ok ? 0 : 1)], 1u);
}


int bc_expand(int v, int bits)
{
   return (v << (8 - bits)) | (v >> (2 * bits - 8));
}

/* Decode a whole 4x4 BC7 block: header parsed once, per-texel tail unchanged (bit-exact).
 * Guarded out of variants that cannot reach BC7, to cut pipeline compile time. */
#if defined(BC_OUT_UNORM8) || defined(BC_OUT_BC3) || defined(BC_OUT_ASTC) || defined(BC_OUT_ASTC7)
#if !defined(BC_TABLES_BUFFER)

/* Decode a whole 4x4 BC7 block into bc_px, bit-exact with the decoder below.
 *
 * Written so that nothing is indexed by a per-lane value except the six packed endpoints: the
 * mode's parameters come from packed constants, endpoints are read at computed bit offsets into
 * fixed slots (subsets past nsub read junk that is never selected), the index streams are two
 * 64-bit windows read once, weights are computed rather than looked up, and the anchors are three
 * per-block positions (16 = none). A loop bound or array index that depends on the mode makes the
 * compiler turn every access into a select over the whole array. */
void bc7_decode_block(uvec4 blk)
{
   const int mode = findLSB(blk.x & 0xffu);
   if (mode < 0)
      { bc_px = uvec4[4](uvec4(0u), uvec4(0u), uvec4(0u), uvec4(0u)); return; }  /* reserved */

   const uint m4 = uint(mode * 4);
   const int nsub = int((0x95bbu >> uint(mode * 2)) & 3u);
   const int npb = int((0x60006664u >> m4) & 15u);
   const int cbits = int((0x57757564u >> m4) & 15u);
   const int abits = int((0x57860000u >> m4) & 15u);
   const int ib = int((0x24222233u >> m4) & 15u);
   const int ib2 = int((0x00230000u >> m4) & 15u);
   const int has_rot = int((0x30u >> uint(mode)) & 1u);
   const int has_isel = int((0x10u >> uint(mode)) & 1u);
   const int epb = int((0xc9u >> uint(mode)) & 1u);
   const int spb = int((0x02u >> uint(mode)) & 1u);

   int bit = mode + 1;
   const int part = int(bc_bits(blk, bit, npb));
   bit += npb;
   const int rotation = int(bc_bits(blk, bit, 2 * has_rot));
   bit += 2 * has_rot;
   const int isel = int(bc_bits(blk, bit, has_isel));
   bit += has_isel;

   const int abase = bit + 6 * nsub * cbits;
   const int pbase = abase + 2 * nsub * abits;
   const int pext = epb + spb;
   const int ctot = cbits + pext, atot = abits + pext;

   /* Endpoint k = 2 * subset + end, packed RGBA8 after the p-bit and the expansion to 8 bits. */
   /* Every mode's fields of one channel (2 * nsub * cbits <= 30 bits), its alpha fields (<= 20)
    * and its p-bits (<= 6) each fit one 32-bit window: five unaligned reads, then a bit-field
    * extract per field. Offsets are clamped so the junk slots past nsub stay in range. */
   const uint wr = bc_win32(blk, bit), wg = bc_win32(blk, bit + 2 * nsub * cbits);
   const uint wb = bc_win32(blk, bit + 4 * nsub * cbits), wal = bc_win32(blk, abase);
   const uint wp = bc_win32(blk, pbase);
   uint ep[6];
   for (int k = 0; k < 6; k++) {
      const uint p = bitfieldExtract(wp, epb != 0 ? k : (k >> 1), pext);
      const int co = min(k * cbits, 32 - cbits), ao = min(k * abits, 32 - abits);
      uint v = uint(bc_expand(int((bitfieldExtract(wr, co, cbits) << uint(pext)) | p), ctot));
      v |= uint(bc_expand(int((bitfieldExtract(wg, co, cbits) << uint(pext)) | p), ctot)) << 8u;
      v |= uint(bc_expand(int((bitfieldExtract(wb, co, cbits) << uint(pext)) | p), ctot)) << 16u;
      const int qa = int((bitfieldExtract(wal, ao, abits) << uint(pext)) | p);
      v |= (abits > 0 ? uint(bc_expand(qa, atot)) : 255u) << 24u;
      ep[k] = v;
   }

   const uint subsets = nsub == 2 ? bc_part2[part] : (nsub == 3 ? bc_part3[part] : 0u);
   const int ax = nsub == 2 ? bc_anchor2[part] : (nsub == 3 ? bc_anchor3b[part] : 16);
   const int ay = nsub == 3 ? bc_anchor3c[part] : 16;

   /* The index data: primary stream at bit_base (16 * ib - nsub bits), secondary after it. Colour
    * and alpha each read one of them; without a secondary both read the primary. */
   const int bit_base = pbase + (epb != 0 ? 2 * nsub : spb * nsub);
   const int sec_base = bit_base + 16 * ib - nsub;
   const uvec2 s0 = uvec2(bc_win32(blk, bit_base), bc_win32(blk, bit_base + 32));
   const uvec2 s1 = uvec2(bc_win32(blk, sec_base), bc_win32(blk, sec_base + 32));
   const bool c_sec = isel != 0, a_sec = ib2 != 0 && isel == 0;
   const int cib = c_sec ? ib2 : ib, aib = a_sec ? ib2 : ib;
   /* BC7 weights are round(64 * i / (2^n - 1)); no entry is within 0.03 of a half. */
   const float cscale = 64.0 / float((1 << cib) - 1), ascale = 64.0 / float((1 << aib) - 1);
   /* Rotation swaps a colour byte with alpha; 24 swaps alpha with itself. */
   const int rot8 = rotation == 0 ? 24 : 8 * (rotation - 1);

   /* Both streams are consumed in texel order: take the low bits, shift the 64-bit window on. An
    * anchor texel's index is one bit shorter. Without a secondary stream it is read 0 bits wide. */
   uvec2 ps = s0, qs = s1;
   for (int t = 0; t < 16; t++) {
      const int cut = (t == 0 || t == ax || t == ay) ? 1 : 0;
      const int sub = int((subsets >> uint(2 * t)) & 3u);
      const uint ea = sub == 0 ? ep[0] : (sub == 1 ? ep[2] : ep[4]);
      const uint eb = sub == 0 ? ep[1] : (sub == 1 ? ep[3] : ep[5]);
      const int pw = ib - cut, qw = max(ib2 - cut, 0);
      const uint pi = bitfieldExtract(ps.x, 0, pw), qi = bitfieldExtract(qs.x, 0, qw);
      ps = bc_shr64(ps, pw);
      qs = bc_shr64(qs, qw);
      const int wc = int(float(c_sec ? qi : pi) * cscale + 0.5);
      const int wa = int(float(a_sec ? qi : pi) * ascale + 0.5);

      /* a * (64 - w) + b * w as a * 64 + b * w - a * w on unsigned bytes and a masked weight: both
       * products then provably fit 24 bits and run as full-rate v_mul_u32_u24 instead of the
       * quarter-rate 32-bit multiply a signed (b - a) * w needs. The sum wraps back in range. */
      /* R and B share one register as two 16-bit lanes (every lane value stays below 2^16, and
       * a * 64 >= a * w so the subtraction never borrows across lanes); G and A have their own
       * weights. */
      const uint wcm = uint(wc) & 0x7fu, wam = uint(wa) & 0x7fu;
      const uint arb = ea & 0xff00ffu, brb = eb & 0xff00ffu;
      const uint rb = (((arb << 6u) + brb * wcm + 0x200020u - arb * wcm) >> 6u) & 0xff00ffu;
      const uint ag = (ea >> 8u) & 0xffu, bg = (eb >> 8u) & 0xffu;
      const uint aa = ea >> 24u, ba = eb >> 24u;
      const uint g = ((ag << 6u) + bg * wcm + 32u - ag * wcm) >> 6u;
      const uint al = ((aa << 6u) + ba * wam + 32u - aa * wam) >> 6u;
      uint px = rb | (g << 8u) | (al << 24u);
      const uint sw = bitfieldExtract(px, rot8, 8);
      px = bitfieldInsert(px, px >> 24u, rot8, 8);
      px = bitfieldInsert(px, sw, 24, 8);
      bc_px[t >> 2][t & 3] = px;
   }
}
#else
void bc7_decode_block(uvec4 blk)
{
   int mode = -1;
   for (int i = 0; i < 8; i++) {
      if (bc_bits(blk, i, 1) != 0u) { mode = i; break; }
   }
   if (mode < 0)
      { bc_px = uvec4[4](uvec4(0u), uvec4(0u), uvec4(0u), uvec4(0u)); return; }  /* reserved */

   int bit = mode + 1;
   int nsub = bc7_nsub[mode];
   int part = int(bc_bits(blk, bit, bc7_npb[mode]));
   bit += bc7_npb[mode];

   uint subsets = nsub == 2 ? bc_part2[part] : (nsub == 3 ? bc_part3[part] : 0u);

   int rotation = 0;
   if (bc7_rot[mode] != 0) { rotation = int(bc_bits(blk, bit, 2)); bit += 2; }
   int isel = 0;
   if (bc7_isel[mode] != 0) { isel = int(bc_bits(blk, bit, 1)); bit += 1; }

   ivec4 ep[6];
   int cbits = bc7_cbits[mode];
   int abits = bc7_abits[mode];

   for (int c = 0; c < 3; c++)
      for (int s = 0; s < nsub; s++)
         for (int e = 0; e < 2; e++) {
            ep[s * 2 + e][c] = int(bc_bits(blk, bit, cbits));
            bit += cbits;
         }

   int ncomp = 3;
   if (abits > 0) {
      for (int s = 0; s < nsub; s++)
         for (int e = 0; e < 2; e++) {
            ep[s * 2 + e][3] = int(bc_bits(blk, bit, abits));
            bit += abits;
         }
      ncomp = 4;
   } else {
      for (int s = 0; s < nsub; s++)
         for (int e = 0; e < 2; e++)
            ep[s * 2 + e][3] = 255;
   }

   if (bc7_epb[mode] != 0) {
      for (int s = 0; s < nsub; s++)
         for (int e = 0; e < 2; e++) {
            int p = int(bc_bits(blk, bit, 1));
            bit += 1;
            for (int c = 0; c < ncomp; c++)
               ep[s * 2 + e][c] = (ep[s * 2 + e][c] << 1) | p;
         }
   } else if (bc7_spb[mode] != 0) {
      for (int s = 0; s < nsub; s++) {
         int p = int(bc_bits(blk, bit, 1));
         bit += 1;
         for (int e = 0; e < 2; e++)
            for (int c = 0; c < ncomp; c++)
               ep[s * 2 + e][c] = (ep[s * 2 + e][c] << 1) | p;
      }
   }

   int extra = bc7_epb[mode] + bc7_spb[mode];
   for (int s = 0; s < nsub; s++)
      for (int e = 0; e < 2; e++) {
         for (int c = 0; c < 3; c++)
            ep[s * 2 + e][c] = bc_expand(ep[s * 2 + e][c], cbits + extra);
         if (abits > 0)
            ep[s * 2 + e][3] = bc_expand(ep[s * 2 + e][3], abits + extra);
      }

   /* Per texel. `bit` is the block-constant base of the index data: do not mutate it here. */
   const int ib = bc7_ib[mode];
   const int ib2 = bc7_ib2[mode];
   const int bit_base = bit;
   for (int texel = 0; texel < 16; texel++) {
      int anchors = bc_anchors_before(nsub, part, texel);

      int sec_bit = bit_base + 16 * ib - nsub + ib2 * texel - anchors;
      int tbit = bit_base + ib * texel - anchors;

      int sub = int((subsets >> (texel * 2)) & 3u);
      bool anchor = bc_is_anchor(nsub, part, texel);

      int idx0 = int(bc_bits(blk, tbit, ib - (anchor ? 1 : 0)));
      int idx1 = 0;
      if (ib2 != 0)
         idx1 = int(bc_bits(blk, sec_bit, ib2 - (anchor ? 1 : 0)));

      int cidx = isel != 0 ? idx1 : idx0;
      int cib = isel != 0 ? ib2 : ib;

      /* Select the subset's endpoints once, so the component loop indexes a plain ivec4. */
      ivec4 epa = ep[sub * 2];
      ivec4 epb = ep[sub * 2 + 1];

      ivec4 res;
      for (int c = 0; c < 3; c++)
         res[c] = bc_interpolate(epa[c], epb[c], cidx, cib);

      int aidx, aib;
      if (ib2 != 0 && isel == 0) {
         aidx = idx1;
         aib = ib2;
      } else {
         aidx = idx0;
         aib = ib;
      }
      res[3] = bc_interpolate(epa[3], epb[3], aidx, aib);

      if (rotation != 0) {
         int t = res[rotation - 1];
         res[rotation - 1] = res[3];
         res[3] = t;
      }

      bc_px[texel >> 2][texel & 3] =
         uint(res.r) | (uint(res.g) << 8) | (uint(res.b) << 16) | (uint(res.a) << 24);
   }
}
#endif /* BC_TABLES_BUFFER */
#endif

/* ------------------------------------------------------------------ BC6H */

int bc6_unq_unsigned(int v, int bits)
{
   if (bits >= 15) return v;
   if (v == 0) return 0;
   if (v == (1 << bits) - 1) return 0xffff;
   return ((v << 15) + 0x4000) >> (bits - 1);
}

int bc6_unq_signed(int v, int bits)
{
   if (bits >= 16) return v;
   if (v == 0) return 0;
   bool neg = v < 0;
   if (neg) v = -v;
   if (v >= (1 << (bits - 1)) - 1)
      v = 0x7fff;
   else
      v = ((v << 15) + 0x4000) >> (bits - 1);
   return neg ? -v : v;
}

/* Decode a whole 4x4 BC6H block: mode, field walk, delta expansion and unquantisation run once.
 * Only reachable as BC_OUT_F16 or BC_OUT_RG11B10; guarded out of other variants. */
#if defined(BC_OUT_F16) || defined(BC_OUT_RG11B10)
/* Texel t of the block at c0, unless it is outside the image (the last block of a row or column
 * of an image whose size is not a multiple of 4). h holds the three channels as half-float bits. */
#if defined(BC_OUT_RG11B10)
/* An unsigned half (sign clear) to an 11- or 10-bit unsigned float: the same exponent, the
 * mantissa rounded to nearest even from 10 bits to 10 - drop. A carry into the exponent is the
 * correct rounding; the top finite value is the clamp, never infinity. */
uint bc_half_to_uf(uint h, uint drop, uint maxv)
{
   uint r = (h + ((1u << (drop - 1u)) - 1u) + ((h >> drop) & 1u)) >> drop;
   return min(r, maxv);
}
#endif

#if !defined(BC_TABLES_BUFFER)
/* The decoded texels go to shared memory first, and the workgroup then stores its 32x32 texel
 * tile row by row (bc_tile_store()). Storing them block by block from each invocation scatters
 * every image store of a wave over 64 blocks: measured on the 920, the same stores take 9.7 ms
 * per 2048^2 block by block and 1.6 ms row by row. 17 words per block: an odd stride keeps the
 * 64 lanes' writes on distinct banks. */
#if defined(BC_OUT_RG11B10)
shared uint bc_tile[64 * 17];

void bc_store_texel(ivec3 c0, int bx, int by, int t, uvec3 h)
{
   bc_tile[int(gl_LocalInvocationIndex) * 17 + t] =
      bc_half_to_uf(h.x, 4u, 0x7bfu) | (bc_half_to_uf(h.y, 4u, 0x7bfu) << 11) |
      (bc_half_to_uf(h.z, 5u, 0x3dfu) << 22);
}
#else
shared uvec2 bc_tile[64 * 17];

void bc_store_texel(ivec3 c0, int bx, int by, int t, uvec3 h)
{
   bc_tile[int(gl_LocalInvocationIndex) * 17 + t] = uvec2(h.x | (h.y << 16), h.z | (0x3c00u << 16));
}
#endif

/* Every invocation of the workgroup must call this (it starts with a barrier). Texels outside the
 * region are not stored, so the slots of invocations past the edge are never read for a store. */
void bc_tile_store(int z)
{
   barrier();
   const int lane = int(gl_LocalInvocationIndex);
   const int gx0 = int(gl_WorkGroupID.x) * 32, gy0 = int(gl_WorkGroupID.y) * 32;
   for (int k = 0; k < 16; k++) {
      const int tx = lane & 31, ty = 2 * k + (lane >> 5);
      if (gx0 + tx < pc.extent_x && gy0 + ty < pc.extent_y) {
         const int src = ((ty >> 2) * 8 + (tx >> 2)) * 17 + (ty & 3) * 4 + (tx & 3);
         const ivec3 dst = ivec3(pc.offset.x + gx0 + tx, pc.offset.y + gy0 + ty, z);
#if defined(BC_OUT_RG11B10)
         imageStore(o_img, dst, uvec4(bc_tile[src], 0u, 0u, 0u));
#else
         imageStore(o_img, dst, vec4(unpackHalf2x16(bc_tile[src].x), unpackHalf2x16(bc_tile[src].y)));
#endif
      }
   }
}
#elif defined(BC_OUT_RG11B10)
void bc_store_texel(ivec3 c0, int bx, int by, int t, uvec3 h)
{
   const int lx = t & 3, ly = t >> 2;
   if (bx * 4 + lx < pc.extent_x && by * 4 + ly < pc.extent_y) {
      uint p = bc_half_to_uf(h.x, 4u, 0x7bfu) | (bc_half_to_uf(h.y, 4u, 0x7bfu) << 11) |
               (bc_half_to_uf(h.z, 5u, 0x3dfu) << 22);
      imageStore(o_img, ivec3(c0.x + lx, c0.y + ly, c0.z), uvec4(p, 0u, 0u, 0u));
   }
}
#else
void bc_store_texel(ivec3 c0, int bx, int by, int t, uvec3 h)
{
   const int lx = t & 3, ly = t >> 2;
   if (bx * 4 + lx < pc.extent_x && by * 4 + ly < pc.extent_y)
      imageStore(o_img, ivec3(c0.x + lx, c0.y + ly, c0.z),
                 vec4(unpackHalf2x16(h.x).x, unpackHalf2x16(h.y).x, unpackHalf2x16(h.z).x, 1.0));
}
#endif

#if !defined(BC_TABLES_BUFFER)
/* Decode a whole 4x4 BC6H block, bit-exact with the table-driven decoder below.
 *
 * The endpoint fields are unrolled per mode (generated from the bc6_* tables): each one is read at
 * its constant bit position into a named endpoint, so there is no table load and no per-lane array
 * index. Everything after the header is at a fixed place: two-subset modes have 77 header bits,
 * the partition at 77 and the indices from 82, one-subset modes the indices from 65. The weights
 * are computed, the indices are consumed in texel order, and the interpolation runs on unsigned
 * 16-bit values (signed ones offset by 32768, which floor-divides by 64 unchanged) so its products
 * fit the full-rate 24-bit multiply. */
void bc6h_decode_store(uvec4 blk, bool is_signed, ivec3 c0, int bx, int by)
{
   const uint b0 = blk.x & 0x1fu;
   const int mode = (b0 & 2u) != 0u ? int(((b0 >> 1) & 0xeu) | (b0 & 1u)) + 2 : int(b0 & 1u);

   ivec3 e0 = ivec3(0), e1 = ivec3(0), e2 = ivec3(0), e3 = ivec3(0), dw = ivec3(0);
   int neb = 0;
   bool delta = false;
   switch (mode) {
   case 0:
      neb = 10; dw = ivec3(5, 5, 5); delta = true;
      e2.y |= int(bc_bits(blk, 2, 1) << 4u);
      e2.z |= int(bc_bits(blk, 3, 1) << 4u);
      e3.z |= int(bc_bits(blk, 4, 1) << 4u);
      e0.x |= int(bc_bits(blk, 5, 10));
      e0.y |= int(bc_bits(blk, 15, 10));
      e0.z |= int(bc_bits(blk, 25, 10));
      e1.x |= int(bc_bits(blk, 35, 5));
      e3.y |= int(bc_bits(blk, 40, 1) << 4u);
      e2.y |= int(bc_bits(blk, 41, 4));
      e1.y |= int(bc_bits(blk, 45, 5));
      e3.z |= int(bc_bits(blk, 50, 1));
      e3.y |= int(bc_bits(blk, 51, 4));
      e1.z |= int(bc_bits(blk, 55, 5));
      e3.z |= int(bc_bits(blk, 60, 1) << 1u);
      e2.z |= int(bc_bits(blk, 61, 4));
      e2.x |= int(bc_bits(blk, 65, 5));
      e3.z |= int(bc_bits(blk, 70, 1) << 2u);
      e3.x |= int(bc_bits(blk, 71, 5));
      e3.z |= int(bc_bits(blk, 76, 1) << 3u);
      break;
   case 1:
      neb = 7; dw = ivec3(6, 6, 6); delta = true;
      e2.y |= int(bc_bits(blk, 2, 1) << 5u);
      e3.y |= int(bc_bits(blk, 3, 1) << 4u);
      e3.y |= int(bc_bits(blk, 4, 1) << 5u);
      e0.x |= int(bc_bits(blk, 5, 7));
      e3.z |= int(bc_bits(blk, 12, 1));
      e3.z |= int(bc_bits(blk, 13, 1) << 1u);
      e2.z |= int(bc_bits(blk, 14, 1) << 4u);
      e0.y |= int(bc_bits(blk, 15, 7));
      e2.z |= int(bc_bits(blk, 22, 1) << 5u);
      e3.z |= int(bc_bits(blk, 23, 1) << 2u);
      e2.y |= int(bc_bits(blk, 24, 1) << 4u);
      e0.z |= int(bc_bits(blk, 25, 7));
      e3.z |= int(bc_bits(blk, 32, 1) << 3u);
      e3.z |= int(bc_bits(blk, 33, 1) << 5u);
      e3.z |= int(bc_bits(blk, 34, 1) << 4u);
      e1.x |= int(bc_bits(blk, 35, 6));
      e2.y |= int(bc_bits(blk, 41, 4));
      e1.y |= int(bc_bits(blk, 45, 6));
      e3.y |= int(bc_bits(blk, 51, 4));
      e1.z |= int(bc_bits(blk, 55, 6));
      e2.z |= int(bc_bits(blk, 61, 4));
      e2.x |= int(bc_bits(blk, 65, 6));
      e3.x |= int(bc_bits(blk, 71, 6));
      break;
   case 2:
      neb = 11; dw = ivec3(5, 4, 4); delta = true;
      e0.x |= int(bc_bits(blk, 5, 10));
      e0.y |= int(bc_bits(blk, 15, 10));
      e0.z |= int(bc_bits(blk, 25, 10));
      e1.x |= int(bc_bits(blk, 35, 5));
      e0.x |= int(bc_bits(blk, 40, 1) << 10u);
      e2.y |= int(bc_bits(blk, 41, 4));
      e1.y |= int(bc_bits(blk, 45, 4));
      e0.y |= int(bc_bits(blk, 49, 1) << 10u);
      e3.z |= int(bc_bits(blk, 50, 1));
      e3.y |= int(bc_bits(blk, 51, 4));
      e1.z |= int(bc_bits(blk, 55, 4));
      e0.z |= int(bc_bits(blk, 59, 1) << 10u);
      e3.z |= int(bc_bits(blk, 60, 1) << 1u);
      e2.z |= int(bc_bits(blk, 61, 4));
      e2.x |= int(bc_bits(blk, 65, 5));
      e3.z |= int(bc_bits(blk, 70, 1) << 2u);
      e3.x |= int(bc_bits(blk, 71, 5));
      e3.z |= int(bc_bits(blk, 76, 1) << 3u);
      break;
   case 3:
      neb = 10; dw = ivec3(10, 10, 10); delta = false;
      e0.x |= int(bc_bits(blk, 5, 10));
      e0.y |= int(bc_bits(blk, 15, 10));
      e0.z |= int(bc_bits(blk, 25, 10));
      e1.x |= int(bc_bits(blk, 35, 10));
      e1.y |= int(bc_bits(blk, 45, 10));
      e1.z |= int(bc_bits(blk, 55, 10));
      break;
   case 4:
      neb = 11; dw = ivec3(4, 5, 4); delta = true;
      e0.x |= int(bc_bits(blk, 5, 10));
      e0.y |= int(bc_bits(blk, 15, 10));
      e0.z |= int(bc_bits(blk, 25, 10));
      e1.x |= int(bc_bits(blk, 35, 4));
      e0.x |= int(bc_bits(blk, 39, 1) << 10u);
      e3.y |= int(bc_bits(blk, 40, 1) << 4u);
      e2.y |= int(bc_bits(blk, 41, 4));
      e1.y |= int(bc_bits(blk, 45, 5));
      e0.y |= int(bc_bits(blk, 50, 1) << 10u);
      e3.y |= int(bc_bits(blk, 51, 4));
      e1.z |= int(bc_bits(blk, 55, 4));
      e0.z |= int(bc_bits(blk, 59, 1) << 10u);
      e3.z |= int(bc_bits(blk, 60, 1) << 1u);
      e2.z |= int(bc_bits(blk, 61, 4));
      e2.x |= int(bc_bits(blk, 65, 4));
      e3.z |= int(bc_bits(blk, 69, 1));
      e3.z |= int(bc_bits(blk, 70, 1) << 2u);
      e3.x |= int(bc_bits(blk, 71, 4));
      e2.y |= int(bc_bits(blk, 75, 1) << 4u);
      e3.z |= int(bc_bits(blk, 76, 1) << 3u);
      break;
   case 5:
      neb = 11; dw = ivec3(9, 9, 9); delta = true;
      e0.x |= int(bc_bits(blk, 5, 10));
      e0.y |= int(bc_bits(blk, 15, 10));
      e0.z |= int(bc_bits(blk, 25, 10));
      e1.x |= int(bc_bits(blk, 35, 9));
      e0.x |= int(bc_bits(blk, 44, 1) << 10u);
      e1.y |= int(bc_bits(blk, 45, 9));
      e0.y |= int(bc_bits(blk, 54, 1) << 10u);
      e1.z |= int(bc_bits(blk, 55, 9));
      e0.z |= int(bc_bits(blk, 64, 1) << 10u);
      break;
   case 6:
      neb = 11; dw = ivec3(4, 4, 5); delta = true;
      e0.x |= int(bc_bits(blk, 5, 10));
      e0.y |= int(bc_bits(blk, 15, 10));
      e0.z |= int(bc_bits(blk, 25, 10));
      e1.x |= int(bc_bits(blk, 35, 4));
      e0.x |= int(bc_bits(blk, 39, 1) << 10u);
      e2.z |= int(bc_bits(blk, 40, 1) << 4u);
      e2.y |= int(bc_bits(blk, 41, 4));
      e1.y |= int(bc_bits(blk, 45, 4));
      e0.y |= int(bc_bits(blk, 49, 1) << 10u);
      e3.z |= int(bc_bits(blk, 50, 1));
      e3.y |= int(bc_bits(blk, 51, 4));
      e1.z |= int(bc_bits(blk, 55, 5));
      e0.z |= int(bc_bits(blk, 60, 1) << 10u);
      e2.z |= int(bc_bits(blk, 61, 4));
      e2.x |= int(bc_bits(blk, 65, 4));
      e3.z |= int(bc_bits(blk, 69, 1) << 1u);
      e3.z |= int(bc_bits(blk, 70, 1) << 2u);
      e3.x |= int(bc_bits(blk, 71, 4));
      e3.z |= int(bc_bits(blk, 75, 1) << 4u);
      e3.z |= int(bc_bits(blk, 76, 1) << 3u);
      break;
   case 7:
      neb = 12; dw = ivec3(8, 8, 8); delta = true;
      e0.x |= int(bc_bits(blk, 5, 10));
      e0.y |= int(bc_bits(blk, 15, 10));
      e0.z |= int(bc_bits(blk, 25, 10));
      e1.x |= int(bc_bits(blk, 35, 8));
      e0.x |= int((bitfieldReverse(bc_bits(blk, 43, 2)) >> 30u) << 10u);
      e1.y |= int(bc_bits(blk, 45, 8));
      e0.y |= int((bitfieldReverse(bc_bits(blk, 53, 2)) >> 30u) << 10u);
      e1.z |= int(bc_bits(blk, 55, 8));
      e0.z |= int((bitfieldReverse(bc_bits(blk, 63, 2)) >> 30u) << 10u);
      break;
   case 8:
      neb = 9; dw = ivec3(5, 5, 5); delta = true;
      e0.x |= int(bc_bits(blk, 5, 9));
      e2.z |= int(bc_bits(blk, 14, 1) << 4u);
      e0.y |= int(bc_bits(blk, 15, 9));
      e2.y |= int(bc_bits(blk, 24, 1) << 4u);
      e0.z |= int(bc_bits(blk, 25, 9));
      e3.z |= int(bc_bits(blk, 34, 1) << 4u);
      e1.x |= int(bc_bits(blk, 35, 5));
      e3.y |= int(bc_bits(blk, 40, 1) << 4u);
      e2.y |= int(bc_bits(blk, 41, 4));
      e1.y |= int(bc_bits(blk, 45, 5));
      e3.z |= int(bc_bits(blk, 50, 1));
      e3.y |= int(bc_bits(blk, 51, 4));
      e1.z |= int(bc_bits(blk, 55, 5));
      e3.z |= int(bc_bits(blk, 60, 1) << 1u);
      e2.z |= int(bc_bits(blk, 61, 4));
      e2.x |= int(bc_bits(blk, 65, 5));
      e3.z |= int(bc_bits(blk, 70, 1) << 2u);
      e3.x |= int(bc_bits(blk, 71, 5));
      e3.z |= int(bc_bits(blk, 76, 1) << 3u);
      break;
   case 9:
      neb = 16; dw = ivec3(4, 4, 4); delta = true;
      e0.x |= int(bc_bits(blk, 5, 10));
      e0.y |= int(bc_bits(blk, 15, 10));
      e0.z |= int(bc_bits(blk, 25, 10));
      e1.x |= int(bc_bits(blk, 35, 4));
      e0.x |= int((bitfieldReverse(bc_bits(blk, 39, 6)) >> 26u) << 10u);
      e1.y |= int(bc_bits(blk, 45, 4));
      e0.y |= int((bitfieldReverse(bc_bits(blk, 49, 6)) >> 26u) << 10u);
      e1.z |= int(bc_bits(blk, 55, 4));
      e0.z |= int((bitfieldReverse(bc_bits(blk, 59, 6)) >> 26u) << 10u);
      break;
   case 10:
      neb = 8; dw = ivec3(6, 5, 5); delta = true;
      e0.x |= int(bc_bits(blk, 5, 8));
      e3.y |= int(bc_bits(blk, 13, 1) << 4u);
      e2.z |= int(bc_bits(blk, 14, 1) << 4u);
      e0.y |= int(bc_bits(blk, 15, 8));
      e3.z |= int(bc_bits(blk, 23, 1) << 2u);
      e2.y |= int(bc_bits(blk, 24, 1) << 4u);
      e0.z |= int(bc_bits(blk, 25, 8));
      e3.z |= int(bc_bits(blk, 33, 1) << 3u);
      e3.z |= int(bc_bits(blk, 34, 1) << 4u);
      e1.x |= int(bc_bits(blk, 35, 6));
      e2.y |= int(bc_bits(blk, 41, 4));
      e1.y |= int(bc_bits(blk, 45, 5));
      e3.z |= int(bc_bits(blk, 50, 1));
      e3.y |= int(bc_bits(blk, 51, 4));
      e1.z |= int(bc_bits(blk, 55, 5));
      e3.z |= int(bc_bits(blk, 60, 1) << 1u);
      e2.z |= int(bc_bits(blk, 61, 4));
      e2.x |= int(bc_bits(blk, 65, 6));
      e3.x |= int(bc_bits(blk, 71, 6));
      break;
   case 12:
      neb = 8; dw = ivec3(5, 6, 5); delta = true;
      e0.x |= int(bc_bits(blk, 5, 8));
      e3.z |= int(bc_bits(blk, 13, 1));
      e2.z |= int(bc_bits(blk, 14, 1) << 4u);
      e0.y |= int(bc_bits(blk, 15, 8));
      e2.y |= int(bc_bits(blk, 23, 1) << 5u);
      e2.y |= int(bc_bits(blk, 24, 1) << 4u);
      e0.z |= int(bc_bits(blk, 25, 8));
      e3.y |= int(bc_bits(blk, 33, 1) << 5u);
      e3.z |= int(bc_bits(blk, 34, 1) << 4u);
      e1.x |= int(bc_bits(blk, 35, 5));
      e3.y |= int(bc_bits(blk, 40, 1) << 4u);
      e2.y |= int(bc_bits(blk, 41, 4));
      e1.y |= int(bc_bits(blk, 45, 6));
      e3.y |= int(bc_bits(blk, 51, 4));
      e1.z |= int(bc_bits(blk, 55, 5));
      e3.z |= int(bc_bits(blk, 60, 1) << 1u);
      e2.z |= int(bc_bits(blk, 61, 4));
      e2.x |= int(bc_bits(blk, 65, 5));
      e3.z |= int(bc_bits(blk, 70, 1) << 2u);
      e3.x |= int(bc_bits(blk, 71, 5));
      e3.z |= int(bc_bits(blk, 76, 1) << 3u);
      break;
   case 14:
      neb = 8; dw = ivec3(5, 5, 6); delta = true;
      e0.x |= int(bc_bits(blk, 5, 8));
      e3.z |= int(bc_bits(blk, 13, 1) << 1u);
      e2.z |= int(bc_bits(blk, 14, 1) << 4u);
      e0.y |= int(bc_bits(blk, 15, 8));
      e2.z |= int(bc_bits(blk, 23, 1) << 5u);
      e2.y |= int(bc_bits(blk, 24, 1) << 4u);
      e0.z |= int(bc_bits(blk, 25, 8));
      e3.z |= int(bc_bits(blk, 33, 1) << 5u);
      e3.z |= int(bc_bits(blk, 34, 1) << 4u);
      e1.x |= int(bc_bits(blk, 35, 5));
      e3.y |= int(bc_bits(blk, 40, 1) << 4u);
      e2.y |= int(bc_bits(blk, 41, 4));
      e1.y |= int(bc_bits(blk, 45, 5));
      e3.z |= int(bc_bits(blk, 50, 1));
      e3.y |= int(bc_bits(blk, 51, 4));
      e1.z |= int(bc_bits(blk, 55, 6));
      e2.z |= int(bc_bits(blk, 61, 4));
      e2.x |= int(bc_bits(blk, 65, 5));
      e3.z |= int(bc_bits(blk, 70, 1) << 2u);
      e3.x |= int(bc_bits(blk, 71, 5));
      e3.z |= int(bc_bits(blk, 76, 1) << 3u);
      break;
   case 16:
      neb = 6; dw = ivec3(6, 6, 6); delta = false;
      e0.x |= int(bc_bits(blk, 5, 6));
      e3.y |= int(bc_bits(blk, 11, 1) << 4u);
      e3.z |= int(bc_bits(blk, 12, 1));
      e3.z |= int(bc_bits(blk, 13, 1) << 1u);
      e2.z |= int(bc_bits(blk, 14, 1) << 4u);
      e0.y |= int(bc_bits(blk, 15, 6));
      e2.y |= int(bc_bits(blk, 21, 1) << 5u);
      e2.z |= int(bc_bits(blk, 22, 1) << 5u);
      e3.z |= int(bc_bits(blk, 23, 1) << 2u);
      e2.y |= int(bc_bits(blk, 24, 1) << 4u);
      e0.z |= int(bc_bits(blk, 25, 6));
      e3.y |= int(bc_bits(blk, 31, 1) << 5u);
      e3.z |= int(bc_bits(blk, 32, 1) << 3u);
      e3.z |= int(bc_bits(blk, 33, 1) << 5u);
      e3.z |= int(bc_bits(blk, 34, 1) << 4u);
      e1.x |= int(bc_bits(blk, 35, 6));
      e2.y |= int(bc_bits(blk, 41, 4));
      e1.y |= int(bc_bits(blk, 45, 6));
      e3.y |= int(bc_bits(blk, 51, 4));
      e1.z |= int(bc_bits(blk, 55, 6));
      e2.z |= int(bc_bits(blk, 61, 4));
      e2.x |= int(bc_bits(blk, 65, 6));
      e3.x |= int(bc_bits(blk, 71, 6));
      break;
   default: /* reserved */
      for (int t = 0; t < 16; t++)
         bc_store_texel(c0, bx, by, t, uvec3(0u));
      return;
   }

   const bool two = mode != 3 && mode != 5 && mode != 7 && mode != 9;
   const int nib = two ? 3 : 4;
   const ivec3 emask = ivec3((1 << neb) - 1);
   if (delta) {
      e1 = (e0 + ivec3(bc_sign_extend(e1.x, dw.x), bc_sign_extend(e1.y, dw.y), bc_sign_extend(e1.z, dw.z))) & emask;
      e2 = (e0 + ivec3(bc_sign_extend(e2.x, dw.x), bc_sign_extend(e2.y, dw.y), bc_sign_extend(e2.z, dw.z))) & emask;
      e3 = (e0 + ivec3(bc_sign_extend(e3.x, dw.x), bc_sign_extend(e3.y, dw.y), bc_sign_extend(e3.z, dw.z))) & emask;
   }
   /* Unquantised to 16 bits, then offset into 0..65535 for the unsigned interpolation below. */
   const int bias = is_signed ? 32768 : 0;
   uvec3 q[4];
   for (int e = 0; e < 4; e++) {
      const ivec3 v = e == 0 ? e0 : (e == 1 ? e1 : (e == 2 ? e2 : e3));
      ivec3 u;
      for (int c = 0; c < 3; c++)
         u[c] = is_signed ? bc6_unq_signed(bc_sign_extend(v[c], neb), neb) : bc6_unq_unsigned(v[c], neb);
      q[e] = uvec3(u + bias) & 0xffffu;
   }

   const uint subsets = two ? bc_part2[bc_bits(blk, 77, 5)] : 0u;
   const int ax = two ? bc_anchor2[bc_bits(blk, 77, 5)] : 16;
   uvec2 st = two ? uvec2(bc_win32(blk, 82), bc_win32(blk, 114)) : uvec2(bc_win32(blk, 65), bc_win32(blk, 97));
   const float scale = two ? 64.0 / 7.0 : 64.0 / 15.0;

   for (int t = 0; t < 16; t++) {
      const int w = nib - ((t == 0 || t == ax) ? 1 : 0);
      const uint idx = bitfieldExtract(st.x, 0, w);
      st = bc_shr64(st, w);
      const uint wt = uint(int(float(idx) * scale + 0.5)) & 0x7fu;
      const bool hi = ((subsets >> uint(2 * t)) & 3u) != 0u;
      const uvec3 a = hi ? q[2] : q[0], b = hi ? q[3] : q[1];

      uvec3 res;
      for (int c = 0; c < 3; c++) {
         const int v = int(((a[c] << 6u) + b[c] * wt + 32u - a[c] * wt) >> 6u) - bias;
         /* Scaled to the half range: v * 31 / 64 unsigned, |v| * 31 / 32 with the sign bit
          * signed. Both are non-negative here, so shifts instead of signed divisions. */
         const uint m = uint(abs(v)) * 31u;
         res[c] = is_signed ? ((m >> 5u) | (v < 0 ? 0x8000u : 0u)) : (m >> 6u);
      }
      bc_store_texel(c0, bx, by, t, res);
   }
}
#else
/* Stored texel by texel: a 16-texel array would be on the stack on Mali. */
void bc6h_decode_store(uvec4 blk, bool is_signed, ivec3 c0, int bx, int by)
{
   int mode, bit;
   if (bc_bits(blk, 1, 1) != 0u) {
      /* Mesa: mode = (((block[0] >> 1) & 0xe) | (block[0] & 1)) + 2 */
      mode = int(((bc_bits(blk, 0, 8) >> 1) & 0xeu) | bc_bits(blk, 0, 1)) + 2;
      bit = 5;
   } else {
      mode = int(bc_bits(blk, 0, 2));
      bit = 2;
   }

   if ((bc6_flags[mode] & 1) != 0)   /* reserved mode */
      { for (int t = 0; t < 16; t++) bc_store_texel(c0, bx, by, t, uvec3(0u)); return; }

   int neb = bc6_neb[mode];
   int nib = bc6_nib[mode];
   int npb = bc6_npb[mode];
   int nend = npb != 0 ? 4 : 2;

   ivec4 ep[4];
   for (int e = 0; e < 4; e++)
      ep[e] = ivec4(0);

   for (int f = bc6_foff[mode]; f < bc6_foff[mode + 1]; f++) {
      uint fd = bc6_fields[f];
      int fep = int(fd & 3u);
      int fc = int((fd >> 2) & 3u);
      int foff = int((fd >> 4) & 15u);
      int fnb = int((fd >> 8) & 31u);
      bool rev = ((fd >> 13) & 1u) != 0u;

      int v = int(bc_bits(blk, bit, fnb));
      bit += fnb;

      if (rev) {
         /* Reverse the low fnb bits of v and deposit them at foff. fnb > 0 guards a shift by 32. */
         if (fnb > 0)
            ep[fep][fc] |= int((bitfieldReverse(uint(v)) >> uint(32 - fnb)) << uint(foff));
      } else {
         ep[fep][fc] |= v << foff;
      }
   }

   if ((bc6_flags[mode] & 2) != 0) {   /* endpoints are deltas from endpoint 0 */
      int d[3] = int[](bc6_d0[mode], bc6_d1[mode], bc6_d2[mode]);
      for (int e = 1; e < nend; e++)
         for (int c = 0; c < 3; c++) {
            int v = bc_sign_extend(ep[e][c], d[c]);
            ep[e][c] = (ep[0][c] + v) & ((1 << neb) - 1);
         }
   }

   for (int e = 0; e < nend; e++)
      for (int c = 0; c < 3; c++) {
         if (is_signed)
            ep[e][c] = bc6_unq_signed(bc_sign_extend(ep[e][c], neb), neb);
         else
            ep[e][c] = bc6_unq_unsigned(ep[e][c], neb);
      }

   int part = 0;
   int nsub = 1;
   uint subsets = 0u;
   if (npb != 0) {
      part = int(bc_bits(blk, bit, npb));
      bit += npb;
      subsets = bc_part2[part];
      nsub = 2;
   }

   /* Per texel. `bit` is the block-constant base of the index data: do not mutate it here. */
   const int bit_base = bit;
   for (int texel = 0; texel < 16; texel++) {
      int anchors = bc_anchors_before(nsub, part, texel);
      int tbit = bit_base + nib * texel - anchors;

      int sub = int((subsets >> (texel * 2)) & 3u);
      int index = int(bc_bits(blk, tbit, nib - (bc_is_anchor(nsub, part, texel) ? 1 : 0)));

      ivec4 epa = ep[sub * 2];
      ivec4 epb = ep[sub * 2 + 1];

      uvec3 res;
      for (int c = 0; c < 3; c++) {
         int v = bc_interpolate(epa[c], epb[c], index, nib);
         if (is_signed)
            v = v < 0 ? int(uint(-v * 31 / 32) | 0x8000u) : (v * 31 / 32);
         else
            v = v * 31 / 64;
         res[c] = uint(v) & 0xffffu;
      }
      bc_store_texel(c0, bx, by, texel, res);
   }
}
#endif /* BC_TABLES_BUFFER */
#endif

#if defined(BC_OUT_ASTC)
/* ---- BC1/BC2/BC3/BC7 -> ASTC 4x4, for texture units without any BC decoder ----------------------
 *
 * The Exynos 9820's Mali-G76 decodes no BC format at all; ASTC and EAC it does. The C reference of
 * everything below, and the check that the texture unit decodes each block mode exactly as
 * modelled, are probe/mali/astc_bc.h and astcmode.c. Measured there on real RE7 blocks: BC1 within
 * 2 LSB everywhere, BC2/BC3 colour within 4 LSB and alpha 94% within 10, BC7 colour 85.8% within 10.
 *
 * Block modes (4x4 weight grid):  0x042 single plane, 4 weight levels   0x051 single, 3 levels
 *                                 0x053 single, 8 levels                0x442 dual, 4 levels
 *                                 0x451 dual, 3 levels
 * Layout: mode bits 0-10, one partition, CEM at 13-16, the endpoint ISE from bit 17 upward, the
 * weight ISE from bit 127 downward (bit-reversed stream), CCS just below it for dual plane, and
 * dual-plane weights interleaved plane 0 / plane 1 per texel. Endpoint order r0 r1 g0 g1 b0 b1 a0 a1.
 * Endpoint ranges used: 256 (8 plain bits), 128 (7 bits), 48 (a trit and 4 bits). Direct modes
 * swap (with blue contraction) when e1 sums lower than e0, so the pairs are ordered to prevent it. */
/* Generated by probe/mali/astc_tables.py from the ASTC spec procedures. Bytes, four per uint. */
#if !defined(BC_TABLES_BUFFER)
const uint astc_trit_enc[61] = uint[](0x04020100u, 0x09080605u, 0x1211100au, 0x18161514u, 0x07031a19u, 0x1b17130bu, 0x200e0d0cu, 0x25242221u, 0x2a292826u, 0x34323130u, 0x39383635u, 0x2b27233au, 0x2c3b3733u, 0x41402e2du, 0x46454442u, 0x504a4948u, 0x55545251u, 0x5a595856u, 0x534b4743u, 0x4d4c5b57u, 0x8281804eu, 0x88868584u, 0x91908a89u, 0x96959492u, 0x839a9998u, 0x97938b87u, 0x8e8d8c9bu, 0xa4a2a1a0u, 0xa9a8a6a5u, 0xb2b1b0aau, 0xb8b6b5b4u, 0xa7a3bab9u, 0xbbb7b3abu, 0xc0aeadacu, 0xc5c4c2c1u, 0xcac9c8c6u, 0xd4d2d1d0u, 0xd9d8d6d5u, 0xcbc7c3dau, 0xccdbd7d3u, 0x6160cecdu, 0x66656462u, 0x706a6968u, 0x75747271u, 0x7a797876u, 0x736b6763u, 0x6d6c7b77u, 0xe2e1e06eu, 0xe8e6e5e4u, 0xf1f0eae9u, 0xf6f5f4f2u, 0xe3faf9f8u, 0xf7f3ebe7u, 0xeeedecfbu, 0x3c1e1d1cu, 0x5d5c3e3du, 0x9e9d9c5eu, 0xdcbebdbcu, 0x3f1fdeddu, 0xdfbf9f5fu, 0x007e7d7cu);
const uint astc_q48_unq[12] = uint[](0xef10ff00u, 0xcf30df20u, 0xae51be41u, 0x8e719e61u, 0xea15fa05u, 0xc936d926u, 0xa956b946u, 0x88779867u, 0xe41bf40bu, 0xc43bd42bu, 0xa35cb34cu, 0x837c936cu);
const uint astc_q48_near[64] = uint[](0x10000000u, 0x10101010u, 0x20202010u, 0x02022020u, 0x12020202u, 0x12121212u, 0x22222212u, 0x04042222u, 0x04040404u, 0x14141414u, 0x24242414u, 0x06062424u, 0x06060606u, 0x16161616u, 0x26262616u, 0x08082626u, 0x08080808u, 0x18181818u, 0x28281818u, 0x0a282828u, 0x0a0a0a0au, 0x1a1a1a1au, 0x2a2a1a1au, 0x0c2a2a2au, 0x0c0c0c0cu, 0x1c1c1c0cu, 0x2c2c1c1cu, 0x0e2c2c2cu, 0x0e0e0e0eu, 0x1e1e1e0eu, 0x2e2e1e1eu, 0x2e2e2e2eu, 0x2f2f2f2fu, 0x1f1f2f2fu, 0x0f1f1f1fu, 0x0f0f0f0fu, 0x2d2d2d0fu, 0x1d1d2d2du, 0x0d1d1d1du, 0x0d0d0d0du, 0x2b2b2b0du, 0x1b1b2b2bu, 0x1b1b1b1bu, 0x0b0b0b0bu, 0x2929290bu, 0x19192929u, 0x19191919u, 0x09090909u, 0x27270909u, 0x17272727u, 0x17171717u, 0x07070707u, 0x25250707u, 0x15252525u, 0x15151515u, 0x05050505u, 0x23230505u, 0x13232323u, 0x13131313u, 0x03030313u, 0x21210303u, 0x11212121u, 0x11111111u, 0x01010111u);
const uint astc_q128_near[64] = uint[](0x01010000u, 0x03030202u, 0x05050404u, 0x07070606u, 0x09090808u, 0x0b0b0a0au, 0x0d0d0c0cu, 0x0f0f0e0eu, 0x11111010u, 0x13131212u, 0x15151414u, 0x17171616u, 0x19191818u, 0x1b1b1a1au, 0x1d1d1c1cu, 0x1f1f1e1eu, 0x21212020u, 0x23232222u, 0x25252424u, 0x27272626u, 0x29292828u, 0x2b2b2a2au, 0x2d2d2c2cu, 0x2f2f2e2eu, 0x31313030u, 0x33333232u, 0x35353434u, 0x37373636u, 0x39393838u, 0x3b3b3a3au, 0x3d3d3c3cu, 0x3f3f3e3eu, 0x41404040u, 0x43424241u, 0x45444443u, 0x47464645u, 0x49484847u, 0x4b4a4a49u, 0x4d4c4c4bu, 0x4f4e4e4du, 0x5150504fu, 0x53525251u, 0x55545453u, 0x57565655u, 0x59585857u, 0x5b5a5a59u, 0x5d5c5c5bu, 0x5f5e5e5du, 0x6160605fu, 0x63626261u, 0x65646463u, 0x67666665u, 0x69686867u, 0x6b6a6a69u, 0x6d6c6c6bu, 0x6f6e6e6du, 0x7170706fu, 0x73727271u, 0x75747473u, 0x77767675u, 0x79787877u, 0x7b7a7a79u, 0x7d7c7c7bu, 0x7f7e7e7du);
#endif
int astc_byte(uint w, int i) { return int((w >> uint((i & 3) * 8)) & 0xffu); }
int astc_trit_enc_at(int i) { return astc_byte(astc_trit_enc[i >> 2], i); }
int astc_q48_unq_at(int i) { return astc_byte(astc_q48_unq[i >> 2], i); }
int astc_q48_near_at(int i) { return astc_byte(astc_q48_near[i >> 2], i); }
int astc_q128_near_at(int i) { return astc_byte(astc_q128_near[i >> 2], i); }

/* The block being built. e0 and e1 are the two endpoints per channel, which the ISE writes in the
 * order r0 r1 g0 g1 b0 b1 a0 a1. The weights are packed in the order the ISE writes them, 3 bits
 * each for 8 levels and 2 bits otherwise, plane 0 / plane 1 interleaved per texel for dual plane.
 * No arrays, here or below: Mali keeps any array of more than 8 entries on the stack. */
struct AstcBlk {
   int mode;
   int dual;
   int ccs;
   int cem;
   int ep_range;
   int w_levels;
   ivec4 e0;
   ivec4 e1;
   uvec2 w;
};

/* n <= 8 bits at bit `off` of a 64-bit value, read and replaced. */
uint astc_get64(uvec2 v, int off, int n)
{
   const uint x = off >= 32 ? v.y >> uint(off - 32)
                            : (v.x >> uint(off)) | (off == 0 ? 0u : v.y << uint(32 - off));
   return x & ((1u << uint(n)) - 1u);
}

void astc_set64(inout uvec2 v, int off, int n, uint val)
{
   const uint m = (1u << uint(n)) - 1u;
   val &= m;
   if (off >= 32) {
      v.y = (v.y & ~(m << uint(off - 32))) | (val << uint(off - 32));
   } else {
      v.x = (v.x & ~(m << uint(off))) | (val << uint(off));
      if (off + n > 32)
         v.y = (v.y & ~(m >> uint(32 - off))) | (val >> uint(32 - off));
   }
}

int astc_wbits(int levels)
{
   return levels == 8 ? 3 : 2;
}

int astc_w_get(AstcBlk b, int i)
{
   const int nb = astc_wbits(b.w_levels);
   return int(astc_get64(b.w, i * nb, nb));
}

void astc_w_set(inout AstcBlk b, int i, int v)
{
   const int nb = astc_wbits(b.w_levels);
   astc_set64(b.w, i * nb, nb, uint(v));
}

/* Endpoint k in ISE order. */
int astc_ep(AstcBlk b, int k)
{
   return (k & 1) != 0 ? b.e1[k >> 1] : b.e0[k >> 1];
}

/* n <= 32 bits of v at bit `off` of the block, whose bits there are still 0. */
void astc_put(inout uvec4 o, int off, int n, uint v)
{
   if (n < 32)
      v &= (1u << uint(n)) - 1u;
   const int w = off >> 5;
   const uint s = uint(off & 31);
   const uint lo = v << s;
   const uint hi = s != 0u && int(s) + n > 32 ? v >> (32u - s) : 0u;
   o.x |= w == 0 ? lo : 0u;
   o.y |= w == 1 ? lo : (w == 0 ? hi : 0u);
   o.z |= w == 2 ? lo : (w == 1 ? hi : 0u);
   o.w |= w == 3 ? lo : (w == 2 ? hi : 0u);
}

int astc_ep_unq(int range, int v)
{
   return range == 256 ? v : range == 128 ? ((v << 1) | (v >> 6)) : astc_q48_unq_at(v);
}

int astc_ep_near(int range, int x)
{
   return range == 256 ? x : range == 128 ? astc_q128_near_at(x) : astc_q48_near_at(x);
}

int astc_w_unq(int levels, int v)
{
   if (levels == 3)
      return v * 32;
   int r = levels == 8 ? ((v << 3) | v) : ((v << 4) | (v << 2) | v);
   return r > 32 ? r + 1 : r;
}

/* The endpoint ISE from bit 17: 8 or 7 plain bits per value, or for range 48 a trit and 4 bits,
 * five values per group laid out m0 T[1:0] m1 T[3:2] m2 T[4] m3 T[6:5] m4 T[7] and truncated
 * after the last value. */
void astc_put_ep(inout uvec4 o, AstcBlk b)
{
   const int nep = b.cem == 12 ? 8 : 6;
   if (b.ep_range != 48) {
      const int nb = b.ep_range == 128 ? 7 : 8;
      for (int k = 0; k < nep; k++)
         astc_put(o, 17 + k * nb, nb, uint(astc_ep(b, k)));
      return;
   }
   int pos = 17;
   for (int g = 0; g < nep; g += 5) {
      const int n = min(nep - g, 5);
      int tv = 0;
      for (int i = n - 1; i >= 0; i--)
         tv = tv * 3 + (astc_ep(b, g + i) >> 4);
      const uint T = uint(astc_trit_enc_at(tv));
      for (int i = 0; i < n; i++) {
         /* T bits after value i: from bit 0 2 4 5 7, 2 2 1 2 1 of them. */
         const uint tlo = (0x75420u >> uint(4 * i)) & 15u;
         const uint tn = (0x12122u >> uint(4 * i)) & 15u;
         astc_put(o, pos, 4, uint(astc_ep(b, g + i) & 15));
         astc_put(o, pos + 4, int(tn), (T >> tlo) & ((1u << tn) - 1u));
         pos += 4 + int(tn);
      }
   }
}

/* The weight ISE, bit 0 first, and its length in bits. With 3 levels a weight is a trit alone,
 * so a group of five is the 8 bits of T, truncated to 2 4 5 7 8 bits after 1 2 3 4 5 values. */
int astc_w_stream(AstcBlk b, out uvec2 ws)
{
   const int nw = b.dual != 0 ? 32 : 16;
   if (b.w_levels != 3) {
      const int nbits = nw * astc_wbits(b.w_levels);
      ws = uvec2(b.w.x, nbits >= 64 ? b.w.y : b.w.y & ((1u << uint(nbits - 32)) - 1u));
      return nbits;
   }
   ws = uvec2(0u);
   int pos = 0;
   for (int g = 0; g < nw; g += 5) {
      const int n = min(nw - g, 5);
      int tv = 0;
      for (int i = n - 1; i >= 0; i--)
         tv = tv * 3 + astc_w_get(b, g + i);
      const int nt = int((0x875420u >> uint(4 * n)) & 15u);
      astc_set64(ws, pos, nt, uint(astc_trit_enc_at(tv)));
      pos += nt;
   }
   return pos;
}

uvec4 astc_pack(AstcBlk b)
{
   uvec4 o = uvec4(0u);
   astc_put(o, 0, 11, uint(b.mode));
   astc_put(o, 13, 4, uint(b.cem));
   astc_put_ep(o, b);

   /* The weight stream runs down from bit 127, bit-reversed. */
   uvec2 ws;
   const int wbits = astc_w_stream(b, ws);
   o.w |= bitfieldReverse(ws.x);
   o.z |= bitfieldReverse(ws.y);
   if (b.dual != 0)
      astc_put(o, 128 - wbits - 2, 2, uint(b.ccs));
   return o;
}

int astc_sum3(AstcBlk b, bool hi)
{
   const ivec4 e = hi ? b.e1 : b.e0;
   return astc_ep_unq(b.ep_range, e.r) + astc_ep_unq(b.ep_range, e.g) +
          astc_ep_unq(b.ep_range, e.b);
}

void astc_order(inout AstcBlk b)
{
   if (astc_sum3(b, true) >= astc_sum3(b, false))
      return;
   const ivec4 t = b.e0;
   b.e0 = b.e1;
   b.e1 = t;
   /* Every weight w becomes levels - 1 - w: the complement for 4 and 8 levels, 0 <-> 2 for 3. */
   if (b.w_levels == 3)
      b.w ^= (~b.w & uvec2(0x55555555u)) << 1u;
   else
      b.w = ~b.w;
}

ivec3 astc_565(uint c)
{
   int r = int((c >> 11) & 31u), g = int((c >> 5) & 63u), bl = int(c & 31u);
   return ivec3((r << 3) | (r >> 2), (g << 2) | (g >> 4), (bl << 3) | (bl >> 2));
}

AstcBlk astc_blk_init()
{
   AstcBlk b;
   b.mode = 0;
   b.dual = 0;
   b.ccs = 3;
   b.cem = 8;
   b.ep_range = 256;
   b.w_levels = 4;
   b.e0 = ivec4(0);
   b.e1 = ivec4(0);
   b.w = uvec2(0u);
   return b;
}

/* BC1: a 4-colour block keeps its endpoints and indices; a 3-colour block needs the midpoint, so
 * trits; one with transparent texels also puts alpha on its own plane. `off` is the bit offset of
 * the colour block in blk. The index maps are nibble tables: entry i at bits 4i. */
uvec4 astc_from_bc1(uvec4 blk, int off, bool rgba)
{
   const uint c0 = bc_bits(blk, off, 16), c1 = bc_bits(blk, off + 16, 16);
   const uint idx = bc_bits(blk, off + 32, 32);
   const ivec3 e0 = astc_565(c0), e1 = astc_565(c1);
   AstcBlk b = astc_blk_init();
   const bool uses3 = (idx & (idx >> 1) & 0x55555555u) != 0u;

   if (c0 > c1) {
      b.mode = 0x042;
      for (int t = 0; t < 16; t++)
         astc_w_set(b, t, int((0x2130u >> (4u * ((idx >> uint(2 * t)) & 3u))) & 15u));
   } else if (!(uses3 && rgba)) {
      /* Index 3 of an opaque BC1 is black, off the line: take the palette entry nearest black. */
      const uint black = (e0.r + e0.g + e0.b) <= (e1.r + e1.g + e1.b) ? 0u : 2u;
      const uint map3 = 0x0120u | (black << 12);
      b.mode = 0x051;
      b.w_levels = 3;
      for (int t = 0; t < 16; t++)
         astc_w_set(b, t, int((map3 >> (4u * ((idx >> uint(2 * t)) & 3u))) & 15u));
   } else {
      b.mode = 0x451;
      b.dual = 1;
      b.cem = 12;
      b.ep_range = 128;
      b.w_levels = 3;
      for (int t = 0; t < 16; t++) {
         const uint i = (idx >> uint(2 * t)) & 3u;
         astc_w_set(b, t * 2, int((0x0120u >> (4u * i)) & 15u));
         astc_w_set(b, t * 2 + 1, i == 3u ? 0 : 2);
      }
      b.e0.a = 0;   /* alpha 0 */
      b.e1.a = 127; /* alpha 255 */
   }
   for (int c = 0; c < 3; c++) {
      b.e0[c] = astc_ep_near(b.ep_range, e0[c]);
      b.e1[c] = astc_ep_near(b.ep_range, e1[c]);
   }
   astc_order(b);
   return astc_pack(b);
}

/* BC2/BC3: the colour half (always four colours) on plane 0, the 16 alpha values on plane 1. */
uvec4 astc_from_bc23(uvec4 blk, bool bc3)
{
   int pal[8] = int[](0, 0, 0, 0, 0, 0, 0, 0);
   if (bc3) {
      const int a0 = int(bc_bits(blk, 0, 8)), a1 = int(bc_bits(blk, 8, 8));
      pal[0] = a0;
      pal[1] = a1;
      if (a0 > a1) {
         for (int i = 1; i < 7; i++)
            pal[i + 1] = ((7 - i) * a0 + i * a1 + 3) / 7;
      } else {
         for (int i = 1; i < 5; i++)
            pal[i + 1] = ((5 - i) * a0 + i * a1 + 2) / 5;
         pal[6] = 0;
         pal[7] = 255;
      }
   }

   const uint c0 = bc_bits(blk, 64, 16), c1 = bc_bits(blk, 80, 16), idx = bc_bits(blk, 96, 32);
   const ivec3 e0 = astc_565(c0), e1 = astc_565(c1);
   AstcBlk b = astc_blk_init();
   b.mode = 0x442;
   b.dual = 1;
   b.cem = 12;
   b.ep_range = 48;
   int amin = 255, amax = 0;
   for (int t = 0; t < 16; t++) {
      astc_w_set(b, t * 2, int((0x2130u >> (4u * ((idx >> uint(2 * t)) & 3u))) & 15u));
      const int a = bc3 ? pal[bc_bits(blk, 16 + 3 * t, 3)] : int(bc_bits(blk, 4 * t, 4)) * 17;
      amin = min(amin, a);
      amax = max(amax, a);
   }
   for (int c = 0; c < 3; c++) {
      b.e0[c] = astc_ep_near(48, e0[c]);
      b.e1[c] = astc_ep_near(48, e1[c]);
   }
   b.e0.a = astc_ep_near(48, amin);
   b.e1.a = astc_ep_near(48, amax);
   const int A0 = astc_ep_unq(48, b.e0.a) * 257, A1 = astc_ep_unq(48, b.e1.a) * 257;
   for (int t = 0; t < 16; t++) {
      const int a = bc3 ? pal[bc_bits(blk, 16 + 3 * t, 3)] : int(bc_bits(blk, 4 * t, 4)) * 17;
      int best = 0, bd = 1 << 30;
      for (int q = 0; q < 4; q++) {
         const int w = astc_w_unq(4, q);
         const int d = abs(((A0 * (64 - w) + A1 * w + 32) >> 6) - a * 257);
         if (d < bd) {
            bd = d;
            best = q;
         }
      }
      astc_w_set(b, t * 2 + 1, best);
   }
   astc_order(b);
   return astc_pack(b);
}

/* Decoded texels (BC7, in bc_px): CEM 8 with 8 weight levels when opaque; with alpha, dual-plane
 * CEM 12 (alpha on its own weights) or single-plane CEM 12 with 8-bit endpoints, whichever fits
 * better. */
void astc_fit_weights(inout AstcBlk b)
{
   ivec4 E0, E1;
   for (int c = 0; c < 3; c++) {
      E0[c] = astc_ep_unq(b.ep_range, b.e0[c]);
      E1[c] = astc_ep_unq(b.ep_range, b.e1[c]);
   }
   E0.a = b.cem == 12 ? astc_ep_unq(b.ep_range, b.e0.a) : 255;
   E1.a = b.cem == 12 ? astc_ep_unq(b.ep_range, b.e1.a) : 255;
   for (int t = 0; t < 16; t++) {
      const ivec4 tx = bc_px_get(t);
      for (int pl = 0; pl < (b.dual != 0 ? 2 : 1); pl++) {
         int best = 0, bd = 1 << 30;
         for (int q = 0; q < b.w_levels; q++) {
            const int w = astc_w_unq(b.w_levels, q);
            int d2 = 0;
            for (int c = 0; c < 4; c++) {
               if (b.dual != 0 && ((c == b.ccs) != (pl == 1)))
                  continue;
               const int d = ((E0[c] * (64 - w) + E1[c] * w + 32) >> 6) - tx[c];
               d2 += d * d;
            }
            if (d2 < bd) {
               bd = d2;
               best = q;
            }
         }
         astc_w_set(b, b.dual != 0 ? t * 2 + pl : t, best);
      }
   }
}

int astc_fit_err(AstcBlk b)
{
   ivec4 E0, E1;
   for (int c = 0; c < 3; c++) {
      E0[c] = astc_ep_unq(b.ep_range, b.e0[c]);
      E1[c] = astc_ep_unq(b.ep_range, b.e1[c]);
   }
   E0.a = b.cem == 12 ? astc_ep_unq(b.ep_range, b.e0.a) : 255;
   E1.a = b.cem == 12 ? astc_ep_unq(b.ep_range, b.e1.a) : 255;
   int e = 0;
   for (int t = 0; t < 16; t++) {
      const ivec4 tx = bc_px_get(t);
      for (int c = 0; c < 4; c++) {
         const int wi = b.dual != 0 ? astc_w_get(b, t * 2 + (c == b.ccs ? 1 : 0)) : astc_w_get(b, t);
         const int w = astc_w_unq(b.w_levels, wi);
         const int v = (E0[c] * 257 * (64 - w) + E1[c] * 257 * w + 32) >> 6;
         const int d = (v >> 8) - tx[c];
         e += d * d;
      }
   }
   return e;
}

uvec4 astc_from_texels()
{
   ivec4 lo = ivec4(255), hi = ivec4(0);
   for (int t = 0; t < 16; t++) {
      lo = min(lo, bc_px_get(t));
      hi = max(hi, bc_px_get(t));
   }
   if (lo.a == 255) {
      AstcBlk b = astc_blk_init();
      b.mode = 0x053;
      b.w_levels = 8;
      b.e0 = ivec4(lo.rgb, 0);
      b.e1 = ivec4(hi.rgb, 0);
      astc_fit_weights(b);
      return astc_pack(b);
   }
   AstcBlk best = astc_blk_init();
   int be = 1 << 30;
   for (int dual = 0; dual < 2; dual++) {
      AstcBlk b = astc_blk_init();
      b.mode = dual != 0 ? 0x442 : 0x042;
      b.dual = dual;
      b.cem = 12;
      b.ep_range = dual != 0 ? 48 : 256;
      for (int c = 0; c < 4; c++) {
         b.e0[c] = astc_ep_near(b.ep_range, lo[c]);
         b.e1[c] = astc_ep_near(b.ep_range, hi[c]);
      }
      astc_order(b);
      astc_fit_weights(b);
      const int e = astc_fit_err(b);
      if (e < be) {
         be = e;
         best = b;
      }
   }
   return astc_pack(best);
}
#endif

#if defined(BC_OUT_ASTC7)
/* ---- BC7 -> ASTC 4x4 (Xclipse: ASTC LDR is decoded by the texture unit) -----------------------
 *
 * One partition, CEM 8 (opaque blocks) or CEM 12, a 4x4 weight grid, and six candidates per block:
 * single plane with 4, 8 or 16 weight levels, and dual plane (one channel on its own weights) with
 * 3, 4 or 5, each with the largest endpoint range the remaining bits allow. The dual plane channel is
 * the one whose removal leaves the other channels closest to a line. Endpoints start at the
 * principal axis' extremes; the candidates are ranked by an error estimate, and the best two are
 * quantised, given one least-squares endpoint refit and scored. A block of one colour becomes a
 * void-extent block. Measured on the 920 with the RE7 corpus (65536 blocks): 39.1 dB against the
 * exact BC7 decode, where the BC3 carrier gives 32.4 and astcenc -fastest 38.9; the upload takes
 * about 2.2 times the BC3 carrier's GPU time.
 *
 * The C prototype and the experiments that chose the candidates are probe/gfx/astcenc7/astc7.c;
 * the tables below come from astcenc's own quantisation tables (probe/gfx/astcenc7/gen_glsl.py). */
/* Generated by probe/gfx/astcenc7/gen_glsl.py from astcenc's quantization tables. */
/* Endpoint ranges [256, 192, 160, 128, 96, 48, 16]. */
/* Per range and 8-bit value x (index range * 256 + x): the nearest ISE code | its value << 8. */
const uint a7_ep_nq[896] = uint[](0x01010000u, 0x03030202u, 0x05050404u, 0x07070606u, 0x09090808u, 0x0b0b0a0au, 0x0d0d0c0cu, 0x0f0f0e0eu, 0x11111010u, 0x13131212u, 0x15151414u, 0x17171616u, 0x19191818u, 0x1b1b1a1au, 0x1d1d1c1cu, 0x1f1f1e1eu, 0x21212020u, 0x23232222u, 0x25252424u, 0x27272626u, 0x29292828u, 0x2b2b2a2au, 0x2d2d2c2cu, 0x2f2f2e2eu, 0x31313030u, 0x33333232u, 0x35353434u, 0x37373636u, 0x39393838u, 0x3b3b3a3au, 0x3d3d3c3cu, 0x3f3f3e3eu, 0x41414040u, 0x43434242u, 0x45454444u, 0x47474646u, 0x49494848u, 0x4b4b4a4au, 0x4d4d4c4cu, 0x4f4f4e4eu, 0x51515050u, 0x53535252u, 0x55555454u, 0x57575656u, 0x59595858u, 0x5b5b5a5au, 0x5d5d5c5cu, 0x5f5f5e5eu, 0x61616060u, 0x63636262u, 0x65656464u, 0x67676666u, 0x69696868u, 0x6b6b6a6au, 0x6d6d6c6cu, 0x6f6f6e6eu, 0x71717070u, 0x73737272u, 0x75757474u, 0x77777676u, 0x79797878u, 0x7b7b7a7au, 0x7d7d7c7cu, 0x7f7f7e7eu, 0x81818080u, 0x83838282u, 0x85858484u, 0x87878686u, 0x89898888u, 0x8b8b8a8au, 0x8d8d8c8cu, 0x8f8f8e8eu, 0x91919090u, 0x93939292u, 0x95959494u, 0x97979696u, 0x99999898u, 0x9b9b9a9au, 0x9d9d9c9cu, 0x9f9f9e9eu, 0xa1a1a0a0u, 0xa3a3a2a2u, 0xa5a5a4a4u, 0xa7a7a6a6u, 0xa9a9a8a8u, 0xababaaaau, 0xadadacacu, 0xafafaeaeu, 0xb1b1b0b0u, 0xb3b3b2b2u, 0xb5b5b4b4u, 0xb7b7b6b6u, 0xb9b9b8b8u, 0xbbbbbabau, 0xbdbdbcbcu, 0xbfbfbebeu, 0xc1c1c0c0u, 0xc3c3c2c2u, 0xc5c5c4c4u, 0xc7c7c6c6u, 0xc9c9c8c8u, 0xcbcbcacau, 0xcdcdccccu, 0xcfcfceceu, 0xd1d1d0d0u, 0xd3d3d2d2u, 0xd5d5d4d4u, 0xd7d7d6d6u, 0xd9d9d8d8u, 0xdbdbdadau, 0xdddddcdcu, 0xdfdfdedeu, 0xe1e1e0e0u, 0xe3e3e2e2u, 0xe5e5e4e4u, 0xe7e7e6e6u, 0xe9e9e8e8u, 0xebebeaeau, 0xededececu, 0xefefeeeeu, 0xf1f1f0f0u, 0xf3f3f2f2u, 0xf5f5f4f4u, 0xf7f7f6f6u, 0xf9f9f8f8u, 0xfbfbfafau, 0xfdfdfcfcu, 0xfffffefeu, 0x01400000u, 0x04020280u, 0x05420402u, 0x08040682u, 0x09440804u, 0x0c060a84u, 0x0d460c06u, 0x10080e86u, 0x11481008u, 0x140a1288u, 0x154a140au, 0x180c168au, 0x194c180cu, 0x1c0e1a8cu, 0x1d4e1c0eu, 0x20101e8eu, 0x21502010u, 0x24122290u, 0x25522412u, 0x28142692u, 0x29542814u, 0x2c162a94u, 0x2d562c16u, 0x30182e96u, 0x31583018u, 0x341a3298u, 0x355a341au, 0x381c369au, 0x395c381cu, 0x3c1e3a9cu, 0x3d5e3c1eu, 0x40203e9eu, 0x41604020u, 0x442242a0u, 0x45624422u, 0x482446a2u, 0x49644824u, 0x4c264aa4u, 0x4d664c26u, 0x50284ea6u, 0x51685028u, 0x542a52a8u, 0x556a542au, 0x582c56aau, 0x596c582cu, 0x5c2e5aacu, 0x5d6e5c2eu, 0x60305eaeu, 0x61706030u, 0x643262b0u, 0x65726432u, 0x683466b2u, 0x69746834u, 0x6c366ab4u, 0x6d766c36u, 0x70386eb6u, 0x71787038u, 0x743a72b8u, 0x757a743au, 0x783c76bau, 0x797c783cu, 0x7c3e7abcu, 0x7d7e7c3eu, 0x7ebe7ebeu, 0x81bf81bfu, 0x833f827fu, 0x85bd833fu, 0x873d867du, 0x89bb873du, 0x8b3b8a7bu, 0x8db98b3bu, 0x8f398e79u, 0x91b78f39u, 0x93379277u, 0x95b59337u, 0x97359675u, 0x99b39735u, 0x9b339a73u, 0x9db19b33u, 0x9f319e71u, 0xa1af9f31u, 0xa32fa26fu, 0xa5ada32fu, 0xa72da66du, 0xa9aba72du, 0xab2baa6bu, 0xada9ab2bu, 0xaf29ae69u, 0xb1a7af29u, 0xb327b267u, 0xb5a5b327u, 0xb725b665u, 0xb9a3b725u, 0xbb23ba63u, 0xbda1bb23u, 0xbf21be61u, 0xc19fbf21u, 0xc31fc25fu, 0xc59dc31fu, 0xc71dc65du, 0xc99bc71du, 0xcb1bca5bu, 0xcd99cb1bu, 0xcf19ce59u, 0xd197cf19u, 0xd317d257u, 0xd595d317u, 0xd715d655u, 0xd993d715u, 0xdb13da53u, 0xdd91db13u, 0xdf11de51u, 0xe18fdf11u, 0xe30fe24fu, 0xe58de30fu, 0xe70de64du, 0xe98be70du, 0xeb0bea4bu, 0xed89eb0bu, 0xef09ee49u, 0xf187ef09u, 0xf307f247u, 0xf585f307u, 0xf705f645u, 0xf983f705u, 0xfb03fa43u, 0xfd81fb03u, 0xff01fe41u, 0x01200000u, 0x03400120u, 0x04600460u, 0x08020680u, 0x09220802u, 0x0b420922u, 0x0c620c62u, 0x10040e82u, 0x11241004u, 0x13441124u, 0x14641464u, 0x18061684u, 0x19261806u, 0x1b461926u, 0x1c661c66u, 0x20081e86u, 0x21282008u, 0x23482128u, 0x24682468u, 0x280a2688u, 0x292a280au, 0x2b4a292au, 0x2c6a2c6au, 0x300c2e8au, 0x312c300cu, 0x334c312cu, 0x346c346cu, 0x380e368cu, 0x392e380eu, 0x3b4e392eu, 0x3c6e3c6eu, 0x40103e8eu, 0x41304010u, 0x43504130u, 0x44704470u, 0x48124690u, 0x49324812u, 0x4b524932u, 0x4c724c72u, 0x50144e92u, 0x51345014u, 0x53545134u, 0x54745474u, 0x58165694u, 0x59365816u, 0x5b565936u, 0x5c765c76u, 0x60185e96u, 0x61386018u, 0x63586138u, 0x64786478u, 0x681a6698u, 0x693a681au, 0x6b5a693au, 0x6c7a6c7au, 0x701c6e9au, 0x713c701cu, 0x735c713cu, 0x747c747cu, 0x781e769cu, 0x793e781eu, 0x7b5e793eu, 0x7c7e7c7eu, 0x7e9e7e9eu, 0x819f819fu, 0x837f837fu, 0x863f845fu, 0x871f863fu, 0x899d871fu, 0x8b7d8b7du, 0x8e3d8c5du, 0x8f1d8e3du, 0x919b8f1du, 0x937b937bu, 0x963b945bu, 0x971b963bu, 0x9999971bu, 0x9b799b79u, 0x9e399c59u, 0x9f199e39u, 0xa1979f19u, 0xa377a377u, 0xa637a457u, 0xa717a637u, 0xa995a717u, 0xab75ab75u, 0xae35ac55u, 0xaf15ae35u, 0xb193af15u, 0xb373b373u, 0xb633b453u, 0xb713b633u, 0xb991b713u, 0xbb71bb71u, 0xbe31bc51u, 0xbf11be31u, 0xc18fbf11u, 0xc36fc36fu, 0xc62fc44fu, 0xc70fc62fu, 0xc98dc70fu, 0xcb6dcb6du, 0xce2dcc4du, 0xcf0dce2du, 0xd18bcf0du, 0xd36bd36bu, 0xd62bd44bu, 0xd70bd62bu, 0xd989d70bu, 0xdb69db69u, 0xde29dc49u, 0xdf09de29u, 0xe187df09u, 0xe367e367u, 0xe627e447u, 0xe707e627u, 0xe985e707u, 0xeb65eb65u, 0xee25ec45u, 0xef05ee25u, 0xf183ef05u, 0xf363f363u, 0xf623f443u, 0xf703f623u, 0xf981f703u, 0xfb61fb61u, 0xfe21fc41u, 0xff01fe21u, 0x00000000u, 0x02010201u, 0x04020402u, 0x06030603u, 0x08040804u, 0x0a050a05u, 0x0c060c06u, 0x0e070e07u, 0x10081008u, 0x12091209u, 0x140a140au, 0x160b160bu, 0x180c180cu, 0x1a0d1a0du, 0x1c0e1c0eu, 0x1e0f1e0fu, 0x20102010u, 0x22112211u, 0x24122412u, 0x26132613u, 0x28142814u, 0x2a152a15u, 0x2c162c16u, 0x2e172e17u, 0x30183018u, 0x32193219u, 0x341a341au, 0x361b361bu, 0x381c381cu, 0x3a1d3a1du, 0x3c1e3c1eu, 0x3e1f3e1fu, 0x40204020u, 0x42214221u, 0x44224422u, 0x46234623u, 0x48244824u, 0x4a254a25u, 0x4c264c26u, 0x4e274e27u, 0x50285028u, 0x52295229u, 0x542a542au, 0x562b562bu, 0x582c582cu, 0x5a2d5a2du, 0x5c2e5c2eu, 0x5e2f5e2fu, 0x60306030u, 0x62316231u, 0x64326432u, 0x66336633u, 0x68346834u, 0x6a356a35u, 0x6c366c36u, 0x6e376e37u, 0x70387038u, 0x72397239u, 0x743a743au, 0x763b763bu, 0x783c783cu, 0x7a3d7a3du, 0x7c3e7c3eu, 0x7e3f7e3fu, 0x81408140u, 0x83418140u, 0x85428341u, 0x87438542u, 0x89448743u, 0x8b458944u, 0x8d468b45u, 0x8f478d46u, 0x91488f47u, 0x93499148u, 0x954a9349u, 0x974b954au, 0x994c974bu, 0x9b4d994cu, 0x9d4e9b4du, 0x9f4f9d4eu, 0xa1509f4fu, 0xa351a150u, 0xa552a351u, 0xa753a552u, 0xa954a753u, 0xab55a954u, 0xad56ab55u, 0xaf57ad56u, 0xb158af57u, 0xb359b158u, 0xb55ab359u, 0xb75bb55au, 0xb95cb75bu, 0xbb5db95cu, 0xbd5ebb5du, 0xbf5fbd5eu, 0xc160bf5fu, 0xc361c160u, 0xc562c361u, 0xc763c562u, 0xc964c763u, 0xcb65c964u, 0xcd66cb65u, 0xcf67cd66u, 0xd168cf67u, 0xd369d168u, 0xd56ad369u, 0xd76bd56au, 0xd96cd76bu, 0xdb6dd96cu, 0xdd6edb6du, 0xdf6fdd6eu, 0xe170df6fu, 0xe371e170u, 0xe572e371u, 0xe773e572u, 0xe974e773u, 0xeb75e974u, 0xed76eb75u, 0xef77ed76u, 0xf178ef77u, 0xf379f178u, 0xf57af379u, 0xf77bf57au, 0xf97cf77bu, 0xfb7df97cu, 0xfd7efb7du, 0xff7ffd7eu, 0x00000000u, 0x02200220u, 0x05400540u, 0x08020540u, 0x08020802u, 0x0a220a22u, 0x0d420d42u, 0x10040d42u, 0x10041004u, 0x12241224u, 0x15441544u, 0x18061544u, 0x18061806u, 0x1a261a26u, 0x1d461d46u, 0x20081d46u, 0x20082008u, 0x23282328u, 0x25482328u, 0x280a2548u, 0x280a280au, 0x2b2a2b2au, 0x2d4a2b2au, 0x300c2d4au, 0x300c300cu, 0x332c332cu, 0x354c332cu, 0x380e354cu, 0x380e380eu, 0x3b2e3b2eu, 0x3d4e3b2eu, 0x40103d4eu, 0x40104010u, 0x43304330u, 0x46504330u, 0x48124650u, 0x48124812u, 0x4b324b32u, 0x4e524b32u, 0x50144e52u, 0x50145014u, 0x53345334u, 0x56545334u, 0x58165654u, 0x58165816u, 0x5b365b36u, 0x5e565b36u, 0x60185e56u, 0x60186018u, 0x63386338u, 0x66586338u, 0x681a6658u, 0x681a681au, 0x6b3a6b3au, 0x6e5a6b3au, 0x701c6e5au, 0x701c701cu, 0x733c733cu, 0x765c733cu, 0x781e765cu, 0x781e781eu, 0x7b3e7b3eu, 0x7e5e7b3eu, 0x7e5e7e5eu, 0x815f815fu, 0x843f815fu, 0x843f843fu, 0x871f871fu, 0x895d871fu, 0x8c3d895du, 0x8c3d8c3du, 0x8f1d8f1du, 0x915b8f1du, 0x943b915bu, 0x943b943bu, 0x971b971bu, 0x9959971bu, 0x9c399959u, 0x9c399c39u, 0x9f199f19u, 0xa1579f19u, 0xa437a157u, 0xa437a437u, 0xa717a717u, 0xa955a717u, 0xac35a955u, 0xac35ac35u, 0xaf15af15u, 0xb153af15u, 0xb433b153u, 0xb433b433u, 0xb713b713u, 0xb951b713u, 0xbc31b951u, 0xbc31bc31u, 0xbf11bf11u, 0xc24fbf11u, 0xc42fc24fu, 0xc42fc42fu, 0xc70fc70fu, 0xca4dc70fu, 0xcc2dca4du, 0xcc2dcc2du, 0xcf0dcf0du, 0xd24bcf0du, 0xd42bd24bu, 0xd42bd42bu, 0xd70bd70bu, 0xda49d70bu, 0xdc29da49u, 0xdc29dc29u, 0xdf09df09u, 0xe247df09u, 0xe247e247u, 0xe527e527u, 0xe707e707u, 0xea45e707u, 0xea45ea45u, 0xed25ed25u, 0xef05ef05u, 0xf243ef05u, 0xf243f243u, 0xf523f523u, 0xf703f703u, 0xfa41f703u, 0xfa41fa41u, 0xfd21fd21u, 0xff01ff01u, 0x00000000u, 0x05100000u, 0x05100510u, 0x05100510u, 0x0b200510u, 0x0b200b20u, 0x0b200b20u, 0x10021002u, 0x10021002u, 0x15121002u, 0x15121512u, 0x15121512u, 0x1b221512u, 0x1b221b22u, 0x1b221b22u, 0x20042004u, 0x20042004u, 0x20042004u, 0x26142614u, 0x26142614u, 0x2b242614u, 0x2b242b24u, 0x2b242b24u, 0x30063006u, 0x30063006u, 0x30063006u, 0x36163616u, 0x36163616u, 0x3b263616u, 0x3b263b26u, 0x3b263b26u, 0x41084108u, 0x41084108u, 0x41084108u, 0x46184618u, 0x46184618u, 0x46184618u, 0x4c284c28u, 0x4c284c28u, 0x510a4c28u, 0x510a510au, 0x510a510au, 0x561a561au, 0x561a561au, 0x561a561au, 0x5c2a5c2au, 0x5c2a5c2au, 0x610c5c2au, 0x610c610cu, 0x610c610cu, 0x671c610cu, 0x671c671cu, 0x671c671cu, 0x6c2c6c2cu, 0x6c2c6c2cu, 0x710e6c2cu, 0x710e710eu, 0x710e710eu, 0x771e710eu, 0x771e771eu, 0x771e771eu, 0x7c2e7c2eu, 0x7c2e7c2eu, 0x7c2e7c2eu, 0x832f832fu, 0x832f832fu, 0x832f832fu, 0x881f881fu, 0x881f881fu, 0x8e0f881fu, 0x8e0f8e0fu, 0x8e0f8e0fu, 0x932d8e0fu, 0x932d932du, 0x932d932du, 0x981d981du, 0x981d981du, 0x9e0d981du, 0x9e0d9e0du, 0x9e0d9e0du, 0xa32b9e0du, 0xa32ba32bu, 0xa32ba32bu, 0xa91ba91bu, 0xa91ba91bu, 0xa91ba91bu, 0xae0bae0bu, 0xae0bae0bu, 0xb329ae0bu, 0xb329b329u, 0xb329b329u, 0xb919b919u, 0xb919b919u, 0xb919b919u, 0xbe09be09u, 0xbe09be09u, 0xbe09be09u, 0xc427c427u, 0xc427c427u, 0xc917c427u, 0xc917c917u, 0xc917c917u, 0xcf07cf07u, 0xcf07cf07u, 0xcf07cf07u, 0xd425d425u, 0xd425d425u, 0xd915d425u, 0xd915d915u, 0xd915d915u, 0xdf05df05u, 0xdf05df05u, 0xdf05df05u, 0xe423e423u, 0xe423e423u, 0xea13e423u, 0xea13ea13u, 0xea13ea13u, 0xef03ea13u, 0xef03ef03u, 0xef03ef03u, 0xf421f421u, 0xf421f421u, 0xfa11f421u, 0xfa11fa11u, 0xfa11fa11u, 0xff01fa11u, 0xff01ff01u, 0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u, 0x11010000u, 0x11011101u, 0x11011101u, 0x11011101u, 0x11011101u, 0x11011101u, 0x11011101u, 0x11011101u, 0x11011101u, 0x22022202u, 0x22022202u, 0x22022202u, 0x22022202u, 0x22022202u, 0x22022202u, 0x22022202u, 0x22022202u, 0x33032202u, 0x33033303u, 0x33033303u, 0x33033303u, 0x33033303u, 0x33033303u, 0x33033303u, 0x33033303u, 0x33033303u, 0x44044404u, 0x44044404u, 0x44044404u, 0x44044404u, 0x44044404u, 0x44044404u, 0x44044404u, 0x44044404u, 0x55054404u, 0x55055505u, 0x55055505u, 0x55055505u, 0x55055505u, 0x55055505u, 0x55055505u, 0x55055505u, 0x55055505u, 0x66066606u, 0x66066606u, 0x66066606u, 0x66066606u, 0x66066606u, 0x66066606u, 0x66066606u, 0x66066606u, 0x77076606u, 0x77077707u, 0x77077707u, 0x77077707u, 0x77077707u, 0x77077707u, 0x77077707u, 0x77077707u, 0x77077707u, 0x88088808u, 0x88088808u, 0x88088808u, 0x88088808u, 0x88088808u, 0x88088808u, 0x88088808u, 0x88088808u, 0x99098808u, 0x99099909u, 0x99099909u, 0x99099909u, 0x99099909u, 0x99099909u, 0x99099909u, 0x99099909u, 0x99099909u, 0xaa0aaa0au, 0xaa0aaa0au, 0xaa0aaa0au, 0xaa0aaa0au, 0xaa0aaa0au, 0xaa0aaa0au, 0xaa0aaa0au, 0xaa0aaa0au, 0xbb0baa0au, 0xbb0bbb0bu, 0xbb0bbb0bu, 0xbb0bbb0bu, 0xbb0bbb0bu, 0xbb0bbb0bu, 0xbb0bbb0bu, 0xbb0bbb0bu, 0xbb0bbb0bu, 0xcc0ccc0cu, 0xcc0ccc0cu, 0xcc0ccc0cu, 0xcc0ccc0cu, 0xcc0ccc0cu, 0xcc0ccc0cu, 0xcc0ccc0cu, 0xcc0ccc0cu, 0xdd0dcc0cu, 0xdd0ddd0du, 0xdd0ddd0du, 0xdd0ddd0du, 0xdd0ddd0du, 0xdd0ddd0du, 0xdd0ddd0du, 0xdd0ddd0du, 0xdd0ddd0du, 0xee0eee0eu, 0xee0eee0eu, 0xee0eee0eu, 0xee0eee0eu, 0xee0eee0eu, 0xee0eee0eu, 0xee0eee0eu, 0xee0eee0eu, 0xff0fee0eu, 0xff0fff0fu, 0xff0fff0fu, 0xff0fff0fu, 0xff0fff0fu);
/* ISE: five trits / three quints (first value least significant) -> packed T / Q bits. */
const uint a7_trits[61] = uint[](0x04020100u, 0x09080605u, 0x1211100au, 0x18161514u, 0x07031a19u, 0x1b17130fu, 0x200e0d0cu, 0x25242221u, 0x2a292826u, 0x34323130u, 0x39383635u, 0x2f27233au, 0x2c3b3733u, 0x41402e2du, 0x46454442u, 0x504a4948u, 0x55545251u, 0x5a595856u, 0x534f4743u, 0x4d4c5b57u, 0x8281804eu, 0x88868584u, 0x91908a89u, 0x96959492u, 0x839a9998u, 0x97938f87u, 0x8e8d8c9bu, 0xa4a2a1a0u, 0xa9a8a6a5u, 0xb2b1b0aau, 0xb8b6b5b4u, 0xa7a3bab9u, 0xbbb7b3afu, 0xc0aeadacu, 0xc5c4c2c1u, 0xcac9c8c6u, 0xd4d2d1d0u, 0xd9d8d6d5u, 0xcfc7c3dau, 0xccdbd7d3u, 0x6160cecdu, 0x66656462u, 0x706a6968u, 0x75747271u, 0x7a797876u, 0x736f6763u, 0x6d6c7b77u, 0xe2e1e06eu, 0xe8e6e5e4u, 0xf1f0eae9u, 0xf6f5f4f2u, 0xe3faf9f8u, 0xf7f3efe7u, 0xeeedecfbu, 0x3c1e1d1cu, 0x5d5c3e3du, 0x9e9d9c5eu, 0xdcbebdbcu, 0x3f1fdeddu, 0xffbf9f7fu, 0x00fefdfcu);
const uint a7_quints[32] = uint[](0x03020100u, 0x0a090804u, 0x11100c0bu, 0x18141312u, 0x1c1b1a19u, 0x1d150d05u, 0x22212006u, 0x29282423u, 0x302c2b2au, 0x34333231u, 0x3b3a3938u, 0x352d253cu, 0x41400e3du, 0x48444342u, 0x4c4b4a49u, 0x53525150u, 0x5a595854u, 0x4d455c5bu, 0x60165d55u, 0x64636261u, 0x6b6a6968u, 0x7271706cu, 0x79787473u, 0x657c7b7au, 0x1e7d756du, 0x47466766u, 0x4e6f6e26u, 0x77762e4fu, 0x7e365756u, 0x3e5f5e7fu, 0x3f372f27u, 0x0000001fu);
const float a7_ep_ranges[7] = float[](256.0, 192.0, 160.0, 128.0, 96.0, 48.0, 16.0);
const int a7_ep_bits[7] = int[](8, 6, 5, 7, 5, 4, 4);
const int a7_ep_kind[7] = int[](0, 1, 2, 0, 1, 1, 0); /* 0 plain bits, 1 trit, 2 quint */
const int a7_w_levels[6] = int[](3, 4, 5, 8, 12, 16);
const int a7_w_bits[6] = int[](0, 2, 0, 3, 2, 4);
const int a7_w_kind[6] = int[](1, 0, 2, 0, 1, 0);
/* The six candidates: weight level index, and endpoint range index for CEM 12 and for CEM 8. */
const int a7_cand_lv[6] = int[](1, 3, 5, 0, 1, 2);
const int a7_cand_rg12[6] = int[](0, 1, 5, 3, 5, 6);
const int a7_cand_rg8[6] = int[](0, 0, 1, 0, 2, 5);

int a7_byte(uint w, int i) { return int((w >> uint((i & 3) * 8)) & 0xffu); }

/* The tables are read from LDS: in the constant buffer every lookup is a vector memory load, a few
 * hundred per block, and at this shader's occupancy their latency is the encode time. */
shared uint a7_s_ep_nq[a7_ep_nq.length()];
shared uint a7_s_trits[a7_trits.length()];
shared uint a7_s_quints[a7_quints.length()];

/* Called by every invocation of the workgroup, before any of them returns. */
void a7_lds_fill()
{
   const uint l = gl_LocalInvocationIndex, n = gl_WorkGroupSize.x * gl_WorkGroupSize.y;
   for (uint i = l; i < uint(a7_ep_nq.length()); i += n)
      a7_s_ep_nq[i] = a7_ep_nq[i];
   for (uint i = l; i < uint(a7_trits.length()); i += n)
      a7_s_trits[i] = a7_trits[i];
   for (uint i = l; i < uint(a7_quints.length()); i += n)
      a7_s_quints[i] = a7_quints[i];
   barrier();
}

/* The ISE code nearest x in range rg, and that code's value << 8. */
uint a7_nq(int rg, float x)
{
   const int i = rg * 256 + clamp(int(x + 0.5), 0, 255);
   return (a7_s_ep_nq[i >> 1] >> uint((i & 1) * 16)) & 0xffffu;
}

/* e0/e1 are the ISE codes and E0/E1 their values, one byte per channel (RGBA from the low byte), w
 * the sorted weight index of each texel, 4 bits each: plane 0 texel t at nibble t, plane 1 at nibble
 * 16 + t. Kept packed: two of these live across the candidate loop, and registers are what limits
 * this shader's occupancy. */
struct A7 {
   int cem;
   int dual;
   int ccs;
   int lv;
   int rg;
   uint e0;
   uint e1;
   uint E0;
   uint E1;
   uvec4 w;
};

vec4 a7_v4(uint x)
{
   return vec4((uvec4(x) >> uvec4(0u, 8u, 16u, 24u)) & 0xffu);
}

int a7_w_get(A7 b, int k)
{
   return int((b.w[k >> 3] >> uint((k & 7) * 4)) & 15u);
}

void a7_w_set(inout A7 b, int k, int v)
{
   const uint s = uint((k & 7) * 4);
   b.w[k >> 3] = (b.w[k >> 3] & ~(15u << s)) | (uint(v) << s);
}

vec4 a7_mvec(int mask)
{
   return vec4(notEqual(ivec4(mask) & ivec4(1, 2, 4, 8), ivec4(0)));
}

/* Channels on plane p. */
int a7_mask(A7 b, int p)
{
   const int all = b.cem == 12 ? 15 : 7;
   if (b.dual == 0)
      return p == 0 ? all : 0;
   return p == 1 ? (1 << b.ccs) : (all & ~(1 << b.ccs));
}

/* Mean and covariance of the block's four channels, computed once per block. */
vec4 a7_m;
mat4 a7_cov;

void a7_stats()
{
   a7_m = vec4(0.0);
   [[unroll]] for (int t = 0; t < 16; t++)
      a7_m += vec4(bc_px_get(t));
   a7_m *= 1.0 / 16.0;
   a7_cov = mat4(0.0);
   [[unroll]] for (int t = 0; t < 16; t++) {
      const vec4 d = vec4(bc_px_get(t)) - a7_m;
      a7_cov += outerProduct(d, d);
   }
}

/* Principal axis of the channels in mask (three power iterations on the masked covariance), and
 * the summed squared distance of the texels from the line through the mean along it. */
vec4 a7_dir(int mask, out float resid)
{
   const vec4 msk = a7_mvec(mask);
   const mat4 c = mat4(a7_cov[0] * msk * msk.x, a7_cov[1] * msk * msk.y, a7_cov[2] * msk * msk.z,
                       a7_cov[3] * msk * msk.w);
   vec4 a = (vec4(c[0][0], c[1][1], c[2][2], c[3][3]) + vec4(1e-3, 2e-3, 3e-3, 4e-3)) * msk;
   for (int it = 0; it < 3; it++) {
      const vec4 n = c * a;
      const float l = dot(n, n);
      if (l < 1e-12)
         break;
      a = n * inversesqrt(l);
   }
   if (dot(a, a) < 1e-12)
      a = 0.5 * msk;
   a = normalize(a);
   resid = c[0][0] + c[1][1] + c[2][2] + c[3][3] - dot(a, c * a);
   return a;
}

/* Endpoints at the extreme projections onto axis a. */
void a7_ends(vec4 a, out vec4 lo, out vec4 hi)
{
   float pmin = 1e30, pmax = -1e30;
   [[unroll]] for (int t = 0; t < 16; t++) {
      const float s = dot(vec4(bc_px_get(t)) - a7_m, a);
      pmin = min(pmin, s);
      pmax = max(pmax, s);
   }
   lo = a7_m + a * pmin;
   hi = a7_m + a * pmax;
}

/* Quantise the endpoints, order them so the decoder does not blue-contract, then give every texel
 * the weight nearest its projection onto the quantised endpoints. With score, also returns the
 * block's squared error as decoded, in 8-bit units, with the weights on the uniform grid: close
 * enough to rank candidates. The dual plane's second plane is one channel, done in scalars. */
float a7_quantise(inout A7 b, vec4 lo, vec4 hi, const bool score)
{
   const bool rgba = b.cem == 12;
   const uvec4 q0 = uvec4(a7_nq(b.rg, lo.r), a7_nq(b.rg, lo.g), a7_nq(b.rg, lo.b), rgba ? a7_nq(b.rg, lo.a) : 0xff00u);
   const uvec4 q1 = uvec4(a7_nq(b.rg, hi.r), a7_nq(b.rg, hi.g), a7_nq(b.rg, hi.b), rgba ? a7_nq(b.rg, hi.a) : 0xff00u);
   const uvec4 c0 = q0 & 0xffu, c1 = q1 & 0xffu, v0 = q0 >> 8u, v1 = q1 >> 8u;
   const bool swap = v1.r + v1.g + v1.b < v0.r + v0.g + v0.b;
   const uint p0 = c0.r | (c0.g << 8u) | (c0.b << 16u) | (c0.a << 24u);
   const uint p1 = c1.r | (c1.g << 8u) | (c1.b << 16u) | (c1.a << 24u);
   const uint P0 = v0.r | (v0.g << 8u) | (v0.b << 16u) | (v0.a << 24u);
   const uint P1 = v1.r | (v1.g << 8u) | (v1.b << 16u) | (v1.a << 24u);
   b.e0 = swap ? p1 : p0;
   b.e1 = swap ? p0 : p1;
   b.E0 = swap ? P1 : P0;
   b.E1 = swap ? P0 : P1;
   const vec4 E0 = a7_v4(b.E0), E1 = a7_v4(b.E1);
   /* ASTC's weight values are within about one 64th of i * 64 / (levels - 1): rounding to that
    * grid picks the nearest weight, or a neighbour where two are almost equally near. */
   const float top = float(a7_w_levels[b.lv] - 1), ws = 1.0 / top;
   const vec4 m0 = a7_mvec(a7_mask(b, 0));
   const vec4 d = (E1 - E0) * m0;
   const float dd = dot(d, d);
   const float inv = dd > 0.0 ? top / dd : 0.0, base = dot(E0, d);
   float e = 0.0;
   [[unroll]] for (int t = 0; t < 16; t++) {
      const vec4 x = vec4(bc_px_get(t));
      const float q = floor(clamp((dot(x, d) - base) * inv, 0.0, top) + 0.5);
      a7_w_set(b, t, int(q));
      if (score) {
         const vec4 r = (mix(E0, E1, q * ws) - x) * m0;
         e += dot(r, r);
      }
   }
   if (b.dual != 0) {
      const int c = b.ccs;
      const float e0 = E0[c], dc = E1[c] - e0;
      const float inv1 = dc != 0.0 ? top / dc : 0.0;
      [[unroll]] for (int t = 0; t < 16; t++) {
         const float x = float(bc_px_get(t)[c]);
         const float q = floor(clamp((x - e0) * inv1, 0.0, top) + 0.5);
         a7_w_set(b, 16 + t, int(q));
         if (score) {
            const float r = e0 + dc * q * ws - x;
            e += r * r;
         }
      }
   }
   return e;
}

/* Least-squares endpoints for the block's current weights, plane 0's channels together and the
 * dual plane's one channel on its own. */
void a7_ls(A7 b, inout vec4 lo, inout vec4 hi)
{
   const vec4 m0 = a7_mvec(a7_mask(b, 0));
   const float ws = 1.0 / float(a7_w_levels[b.lv] - 1);
   float a = 0.0, ab = 0.0, c = 0.0;
   vec4 x0 = vec4(0.0), x1 = vec4(0.0);
   [[unroll]] for (int t = 0; t < 16; t++) {
      const float w1 = float(a7_w_get(b, t)) * ws, w0 = 1.0 - w1;
      const vec4 px = vec4(bc_px_get(t));
      a += w0 * w0;
      ab += w0 * w1;
      c += w1 * w1;
      x0 += w0 * px;
      x1 += w1 * px;
   }
   const float det = a * c - ab * ab;
   if (abs(det) >= 1e-6) {
      lo = mix(lo, clamp((c * x0 - ab * x1) / det, 0.0, 255.0), m0);
      hi = mix(hi, clamp((a * x1 - ab * x0) / det, 0.0, 255.0), m0);
   }
   if (b.dual != 0) {
      const int ch = b.ccs;
      float a1 = 0.0, ab1 = 0.0, c1 = 0.0, y0 = 0.0, y1 = 0.0;
      [[unroll]] for (int t = 0; t < 16; t++) {
         const float w1 = float(a7_w_get(b, 16 + t)) * ws, w0 = 1.0 - w1;
         const float x = float(bc_px_get(t)[ch]);
         a1 += w0 * w0;
         ab1 += w0 * w1;
         c1 += w1 * w1;
         y0 += w0 * x;
         y1 += w1 * x;
      }
      const float det1 = a1 * c1 - ab1 * ab1;
      if (abs(det1) >= 1e-6) {
         lo[ch] = clamp((c1 * y0 - ab1 * y1) / det1, 0.0, 255.0);
         hi[ch] = clamp((a1 * y1 - ab1 * y0) / det1, 0.0, 255.0);
      }
   }
}

/* One least-squares refit of the endpoints for the block's weights, then quantised again and
 * scored. Kept even when it scores worse (rarely, and by 0.01 dB on the corpus): that saves
 * scoring the block twice. */
float a7_refine(inout A7 b)
{
   vec4 lo = a7_v4(b.E0), hi = a7_v4(b.E1);
   a7_ls(b, lo, hi);
   return a7_quantise(b, lo, hi, true);
}

/* ORs the n (at most 64) low bits of v into the 128-bit o at bit off; those bits of o are still 0. */
void a7_put(inout u64vec2 o, int off, int n, uint64_t v)
{
   v &= n >= 64 ? ~0ul : (1ul << uint(n)) - 1ul;
   o.x |= off < 64 ? v << uint(off) : 0ul;
   o.y |= off >= 64 ? v << uint(off - 64) : (off > 0 ? v >> uint(64 - off) : 0ul);
}

/* Field packers: eight 4-bit fields to eight 2-bit or 3-bit fields, four 8-bit fields to four 4-bit
 * or 7-bit fields, and four 4-bit fields spread to every other 4-bit slot. */
uint a7_n2(uint x)
{
   x &= 0x33333333u;
   x = (x | (x >> 2u)) & 0x0f0f0f0fu;
   x = (x | (x >> 4u)) & 0x00ff00ffu;
   return (x | (x >> 8u)) & 0xffffu;
}

uint a7_n3(uint x)
{
   x &= 0x77777777u;
   x = (x & 0x07070707u) | ((x & 0x70707070u) >> 1u);
   x = (x & 0x003f003fu) | ((x & 0x3f003f00u) >> 2u);
   return (x & 0xfffu) | ((x & 0x0fff0000u) >> 4u);
}

uint a7_b4(uint x)
{
   x &= 0x0f0f0f0fu;
   x = (x | (x >> 4u)) & 0x00ff00ffu;
   return (x | (x >> 8u)) & 0xffffu;
}

uint a7_b7(uint x)
{
   return (x & 0x7fu) | ((x >> 1u) & 0x3f80u) | ((x >> 2u) & 0x1fc000u) | ((x >> 3u) & 0xfe00000u);
}

uint a7_spread(uint x)
{
   x &= 0xffffu;
   x = (x | (x << 8u)) & 0x00ff00ffu;
   return (x | (x << 4u)) & 0x0f0f0f0fu;
}

/* One ISE trit group: cnt (1-5) values with nb plain bits each, laid out m0 T[1:0] m1 T[3:2] m2 T[4]
 * m3 T[6:5] m4 T[7] and cut after the last value's trit bits. Missing values are passed as 0. */
uint64_t a7_tgroup(uint v0, uint v1, uint v2, uint v3, uint v4, int cnt, int nb, out int len)
{
   const uint n = uint(nb);
   const uint tv = (v0 >> n) + 3u * (v1 >> n) + 9u * (v2 >> n) + 27u * (v3 >> n) + 81u * (v4 >> n);
   const uint64_t T = uint64_t(a7_byte(a7_s_trits[tv >> 2], int(tv)));
   const uint64_t m = (1ul << n) - 1ul;
   len = cnt * nb + int((0x87542u >> uint(4 * (cnt - 1))) & 15u);
   return (uint64_t(v0) & m) | ((T & 3ul) << n) | ((uint64_t(v1) & m) << (n + 2u)) |
          (((T >> 2u) & 3ul) << (2u * n + 2u)) | ((uint64_t(v2) & m) << (2u * n + 4u)) |
          (((T >> 4u) & 1ul) << (3u * n + 4u)) | ((uint64_t(v3) & m) << (3u * n + 5u)) |
          (((T >> 5u) & 3ul) << (4u * n + 5u)) | ((uint64_t(v4) & m) << (4u * n + 7u)) |
          ((T >> 7u) << (5u * n + 7u));
}

/* One ISE quint group: m0 Q[2:0] m1 Q[4:3] m2 Q[6:5], cut likewise. */
uint64_t a7_qgroup(uint v0, uint v1, uint v2, int cnt, int nb, out int len)
{
   const uint n = uint(nb);
   const uint qv = (v0 >> n) + 5u * (v1 >> n) + 25u * (v2 >> n);
   const uint64_t Q = uint64_t(a7_byte(a7_s_quints[qv >> 2], int(qv)));
   const uint64_t m = (1ul << n) - 1ul;
   len = cnt * nb + int((0x753u >> uint(4 * (cnt - 1))) & 15u);
   return (uint64_t(v0) & m) | ((Q & 7ul) << n) | ((uint64_t(v1) & m) << (n + 3u)) |
          (((Q >> 3u) & 3ul) << (2u * n + 3u)) | ((uint64_t(v2) & m) << (2u * n + 5u)) |
          ((Q >> 5u) << (3u * n + 5u));
}

uint a7_nib(uvec4 I, int k)
{
   return (I[k >> 3] >> uint((k & 7) * 4)) & 15u;
}

/* Packs the block. The ISE streams are built from the stored fields with a few masks and shifts
 * per stream, or a whole trit or quint group at a time, instead of value by value: the lanes of a
 * wave pack blocks of different shapes, and every shape's path is executed by the whole wave.
 * Weight levels 4, 8 and 16 (plain bits) and 3 and 5 (a trit or a quint alone) are the ones
 * a7_encode() produces; their ISE codes are the sorted indices themselves. */
uvec4 a7_pack(A7 b)
{
   /* Block mode of a 4x4 weight grid: weight range R (3 bits) and precision bit H from the level,
    * D = dual plane. */
   const int L = a7_w_levels[b.lv];
   const int q = L == 3 ? 1 : L == 4 ? 2 : L == 5 ? 3 : L == 8 ? 5 : L == 12 ? 7 : 8;
   const int H = q >= 6 ? 1 : 0, R = q % 6 + 2;
   const int mode = ((R >> 1) & 3) | ((R & 1) << 4) | (2 << 5) | (H << 9) | (b.dual << 10);
   u64vec2 o = u64vec2(uint64_t(mode) | (uint64_t(b.cem) << 13u), 0ul);

   /* Endpoints from bit 17, in ISE order r0 r1 g0 g1 b0 b1 a0 a1. */
   /* Interleave the endpoints' bytes: e0.r e1.r e0.g e1.g | e0.b e1.b e0.a e1.a. */
   const uint lo0 = (b.e0 & 0xffu) | ((b.e0 & 0xff00u) << 8u), lo1 = (b.e1 & 0xffu) | ((b.e1 & 0xff00u) << 8u);
   const uint hi0 = ((b.e0 >> 16u) & 0xffu) | ((b.e0 >> 8u) & 0xff0000u);
   const uint hi1 = ((b.e1 >> 16u) & 0xffu) | ((b.e1 >> 8u) & 0xff0000u);
   const uvec2 ev = uvec2(lo0 | (lo1 << 8u), hi0 | (hi1 << 8u));
   const int n = b.cem == 12 ? 8 : 6, nb = a7_ep_bits[b.rg], kind = a7_ep_kind[b.rg];
   const uint e0 = ev.x & 0xffu, e1 = (ev.x >> 8) & 0xffu, e2 = (ev.x >> 16) & 0xffu, e3 = ev.x >> 24;
   const uint e4 = ev.y & 0xffu, e5 = (ev.y >> 8) & 0xffu, e6 = (ev.y >> 16) & 0xffu, e7 = ev.y >> 24;
   if (kind == 0) {
      uint64_t sv;
      if (nb == 8)
         sv = packUint2x32(ev);
      else if (nb == 7)
         sv = uint64_t(a7_b7(ev.x)) | (uint64_t(a7_b7(ev.y)) << 28u);
      else
         sv = uint64_t(a7_b4(ev.x) | (a7_b4(ev.y) << 16u));
      a7_put(o, 17, n * nb, sv);
   } else if (kind == 1) {
      int l0, l1;
      const uint64_t g0 = a7_tgroup(e0, e1, e2, e3, e4, 5, nb, l0);
      const uint64_t g1 = a7_tgroup(e5, n == 8 ? e6 : 0u, n == 8 ? e7 : 0u, 0u, 0u, n - 5, nb, l1);
      a7_put(o, 17, l0, g0);
      a7_put(o, 17 + l0, l1, g1);
   } else {
      int l0, l1, l2;
      const uint64_t g0 = a7_qgroup(e0, e1, e2, 3, nb, l0);
      const uint64_t g1 = a7_qgroup(e3, e4, e5, 3, nb, l1);
      const uint64_t g2 = a7_qgroup(e6, e7, 0u, 2, nb, l2);
      a7_put(o, 17, l0, g0);
      a7_put(o, 17 + l0, l1, g1);
      if (n == 8)
         a7_put(o, 17 + l0 + l1, l2, g2);
   }

   /* Weights: the stream is built bit 0 first and placed from bit 127 down, bit-reversed. */
   u64vec2 ws = u64vec2(0ul);
   int wbits;
   if (b.dual == 0) {
      if (L == 16) {
         ws.x = packUint2x32(b.w.xy);
         wbits = 64;
      } else if (L == 8) {
         ws.x = uint64_t(a7_n3(b.w.x)) | (uint64_t(a7_n3(b.w.y)) << 24u);
         wbits = 48;
      } else {
         ws.x = uint64_t(a7_n2(b.w.x) | (a7_n2(b.w.y) << 16u));
         wbits = 32;
      }
   } else {
      /* Texel order with plane 0 / plane 1 interleaved: 4-bit fields, eight per word. */
      const uvec4 I = uvec4(a7_spread(b.w.x) | (a7_spread(b.w.z) << 4u),
                            a7_spread(b.w.x >> 16u) | (a7_spread(b.w.z >> 16u) << 4u),
                            a7_spread(b.w.y) | (a7_spread(b.w.w) << 4u),
                            a7_spread(b.w.y >> 16u) | (a7_spread(b.w.w >> 16u) << 4u));
      if (L == 4) {
         ws.x = packUint2x32(uvec2(a7_n2(I.x) | (a7_n2(I.y) << 16u), a7_n2(I.z) | (a7_n2(I.w) << 16u)));
         wbits = 64;
      } else if (L == 3) {
         /* Trits alone: seven groups of 8 bits, the last (two values) cut to 4. */
         int pos = 0;
         [[unroll]] for (int g = 0; g < 32; g += 5) {
            int len;
            const uint64_t c = a7_tgroup(a7_nib(I, g), a7_nib(I, g + 1), g + 2 < 32 ? a7_nib(I, g + 2) : 0u,
                                         g + 3 < 32 ? a7_nib(I, g + 3) : 0u, g + 4 < 32 ? a7_nib(I, g + 4) : 0u,
                                         min(32 - g, 5), 0, len);
            a7_put(ws, pos, len, c);
            pos += len;
         }
         wbits = pos;
      } else {
         /* Quints alone: eleven groups of 7 bits, the last (two values) cut to 5. */
         int pos = 0;
         [[unroll]] for (int g = 0; g < 32; g += 3) {
            int len;
            const uint64_t c = a7_qgroup(a7_nib(I, g), a7_nib(I, g + 1), g + 2 < 32 ? a7_nib(I, g + 2) : 0u,
                                         min(32 - g, 3), 0, len);
            a7_put(ws, pos, len, c);
            pos += len;
         }
         wbits = pos;
      }
   }
   if (b.dual != 0)
      a7_put(o, 128 - wbits - 2, 2, uint64_t(b.ccs));
   const uvec2 w0 = unpackUint2x32(ws.x), w1 = unpackUint2x32(ws.y);
   return uvec4(unpackUint2x32(o.x), unpackUint2x32(o.y)) |
          uvec4(0u, bitfieldReverse(w1.x), bitfieldReverse(w0.y), bitfieldReverse(w0.x));
}

/* The decoded BC7 block (bc_px) as one ASTC 4x4 block. */
uvec4 a7_encode()
{
   bool opaque = true;
   [[unroll]] for (int t = 0; t < 16; t++)
      opaque = opaque && bc_px_get(t).a == 255;
   const int cem = opaque ? 8 : 12, nch = opaque ? 3 : 4, all = opaque ? 7 : 15;

   /* A block of one colour is ASTC's void-extent block (mode bits 0x1fc, LDR, no extent, the
    * colour as four UNORM16 values): exact, and no search. */
   bool one = true;
   [[unroll]] for (int t = 1; t < 16; t++)
      one = one && bc_px[t >> 2][t & 3] == bc_px[0].x;
   if (one) {
      const uvec4 c = ((uvec4(bc_px[0].x) >> uvec4(0u, 8u, 16u, 24u)) & 0xffu) * 257u;
      return uvec4(0xfffffdfcu, 0xffffffffu, c.r | (c.g << 16u), c.b | (c.a << 16u));
   }

   a7_stats();
   float r, rs;
   vec4 slo, shi;
   a7_ends(a7_dir(all, rs), slo, shi);
   /* The dual plane channel: the one whose removal leaves the others closest to a line. */
   int ccs = 0;
   float br = 1e30;
   for (int c = 0; c < nch; c++) {
      a7_dir(all & ~(1 << c), r);
      if (r < br) {
         br = r;
         ccs = c;
      }
   }
   vec4 dlo, dhi;
   a7_ends(a7_dir(all & ~(1 << ccs), r), dlo, dhi);
   int cmin = 255, cmax = 0;
   [[unroll]] for (int t = 0; t < 16; t++) {
      cmin = min(cmin, bc_px_get(t)[ccs]);
      cmax = max(cmax, bc_px_get(t)[ccs]);
   }
   dlo[ccs] = float(cmin);
   dhi[ccs] = float(cmax);

   /* Rank the candidates by an estimate of their error, without quantising anything: the texels'
    * distance from the shape's lines, plus uniform quantisation noise of the weights (scaled by the
    * squared endpoint span, and by 0.4, fitted on the corpus) and of the endpoints. Only the best
    * two are quantised and refitted; on the RE7 corpus this loses nothing against scoring all six. */
   const vec4 dm = a7_mvec(all & ~(1 << ccs));
   const float span_s = dot(shi - slo, shi - slo);
   const float span_d = dot((dhi - dlo) * dm, (dhi - dlo) * dm) + float((cmax - cmin) * (cmax - cmin));
   int k0 = 0, k1 = 1;
   float s0 = 1e30, s1 = 1e30;
   [[unroll]] for (int k = 0; k < 6; k++) {
      const bool dual = k >= 3;
      const float dw = 1.0 / float(a7_w_levels[a7_cand_lv[k]] - 1);
      const float qs = 255.0 / (a7_ep_ranges[opaque ? a7_cand_rg8[k] : a7_cand_rg12[k]] - 1.0);
      const float est = (dual ? br : rs) + (0.4 * 16.0 / 12.0) * (dual ? span_d : span_s) * dw * dw +
                        (16.0 / 12.0 * 2.0 / 3.0) * float(nch) * qs * qs;
      if (est < s0) {
         k1 = k0;
         s1 = s0;
         k0 = k;
         s0 = est;
      } else if (est < s1) {
         k1 = k;
         s1 = est;
      }
   }

   A7 best;
   float be = 1e30;
   [[dont_unroll]] for (int i = 0; i < 2; i++) {
      const int k = i == 0 ? k0 : k1;
      A7 b;
      b.cem = cem;
      b.dual = k >= 3 ? 1 : 0;
      b.ccs = ccs;
      b.lv = a7_cand_lv[k];
      b.rg = opaque ? a7_cand_rg8[k] : a7_cand_rg12[k];
      b.e0 = 0u;
      b.e1 = 0u;
      b.E0 = 0u;
      b.E1 = 0u;
      b.w = uvec4(0u);
      a7_quantise(b, b.dual != 0 ? dlo : slo, b.dual != 0 ? dhi : shi, false);
      const float e = a7_refine(b);
      if (e < be) {
         be = e;
         best = b;
      }
   }
   return a7_pack(best);
}
#endif

/* ------------------------------------------------------------------- main */

void main()
{
#if defined(BC_OUT_ASTC7)
   /* Before any invocation can return: the fill ends in a workgroup barrier. */
   a7_lds_fill();
#endif
   /* One invocation per 4x4 block; radv_meta_bc_decode.c sizes the dispatch in blocks.
    * Skip out-of-range invocations: a zero block has no BC7 mode bit and would read the mode
    * tables out of bounds, corrupting other lanes. */
   ivec2 nblocks = ivec2((pc.extent_x + 3) >> 2, (pc.extent_y + 3) >> 2);
#if (defined(BC_OUT_F16) || defined(BC_OUT_RG11B10)) && !defined(BC_TABLES_BUFFER)
   /* Every invocation must reach the barrier in bc_tile_store(): those past the edge decode a
    * copy of the last block, whose texels are never stored. */
   int bx = min(int(gl_GlobalInvocationID.x), nblocks.x - 1);
   int by = min(int(gl_GlobalInvocationID.y), nblocks.y - 1);
#else
   if (gl_GlobalInvocationID.x >= uint(nblocks.x) || gl_GlobalInvocationID.y >= uint(nblocks.y))
      return;

   int bx = int(gl_GlobalInvocationID.x);
   int by = int(gl_GlobalInvocationID.y);
#endif
   int layer = int(gl_GlobalInvocationID.z);

   /* Top-left texel of this block. pc.offset is block-aligned, so >>2 gives the block index. */
   ivec3 c0 = ivec3(pc.offset.x + bx * 4, pc.offset.y + by * 4, pc.offset.z + layer);
#if defined(BC_SRC_BUFFER)
   const uint src_idx = uint(layer) * pc.src_slice_blocks + uint(by) * pc.src_row_blocks + uint(bx);
#if defined(BC_OUT_EAC_R)
   uvec4 blk = uvec4(BcSrcBlocks8(pc.src_va).b[src_idx], 0u, 0u);
#else
   uvec4 blk = BcSrcBlocks(pc.src_va).b[src_idx];
#endif
#else
   uvec4 blk = texelFetch(s_blocks, ivec3(c0.x >> 2, c0.y >> 2, c0.z), 0);
#endif

   /* [BCCONTENT]: sample 64 source blocks per dispatch for the content-repeat census.
    * Unsynchronised reads can show false distinct keys, never false matches. */
   if (pc.hash_slot >= 0) {
      const int total = nblocks.x * nblocks.y;
      const int idx = by * nblocks.x + bx;
      const int stride = max(1, total / 64);
      if (idx % stride == 0) {
         const int k = idx / stride;
         if (k < 64) {
            uint h = blk.x * 0x9e3779b9u;
            h ^= blk.y * 0x85ebca6bu;
            h ^= blk.z * 0xc2b2ae35u;
            h ^= blk.w * 0x27d4eb2fu;
            h ^= uint(idx) * 0x165667b1u;
            h ^= h >> 15;
            hist.bucket[BC_HASH_BASE + uint(pc.hash_slot) * 64u + uint(k)] = h;
         }
      }
   }

   /* 1 in 64 blocks feeds the histogram (endpoints are bytes 0-1, already read).
    * Skipped on the transcode path, which must stay ALU-free. */
#if !defined(BC_OUT_U32X4) && !defined(BC_OUT_EAC) && !defined(BC_OUT_BC3) && \
    !defined(BC_OUT_ASTC) && !defined(BC_OUT_EAC_R) && !defined(BC_OUT_ASTC7)
   if (((bx | by) & 7) == 0) {
      if (pc.format == BC_FMT_BC4_UNORM || pc.format == BC_FMT_BC4_SNORM) {
         bc_record_spread(blk, 0, int(bc_bits(blk, 0, 8)), int(bc_bits(blk, 8, 8)));
      } else if (pc.format == BC_FMT_BC5_UNORM || pc.format == BC_FMT_BC5_SNORM) {
         int r0 = int(bc_bits(blk, 0, 8)), r1 = int(bc_bits(blk, 8, 8));
         int g0 = int(bc_bits(blk, 64, 8)), g1 = int(bc_bits(blk, 72, 8));
         bc_record_spread(blk, 0, r0, r1);
         bc_record_spread(blk, 64, g0, g1);
         bc_record_rg(r0, r1, g0, g1);
      } else if (pc.format == BC_FMT_BC7) {
         bc_record_bc7(blk);
      }
   }
#endif

#if defined(BC_OUT_U32X4)
   /* BC4 -> BC3, no decode: BC3's alpha half is a BC4 block, so the hardware decodes it.
    * The colour half is zeroed; the view swizzles R <- A (radv_image_view.c).
    * The BC3 alpha unit is off by up to 2/255 on ~23% of texels.
    * BC4_SNORM is not routed here: the alpha block decodes as unorm. */
   imageStore(o_img, ivec3(c0.x >> 2, c0.y >> 2, c0.z), uvec4(blk.x, blk.y, 0u, 0u));
#elif defined(BC_OUT_EAC)
   /* BC5 -> EAC_R11G11, no decode: two BC4-channel re-encodes via lookup (see bc_eac_encode()).
    * The plane stays compressed at 16 bytes per block. BC5_SNORM goes to EAC_R11G11_SNORM. */
   const bool snorm = pc.format == BC_FMT_BC5_SNORM;
   uvec2 r = bc_eac_encode(blk.xy, snorm);
   uvec2 g = bc_eac_encode(blk.zw, snorm);
   imageStore(o_img, ivec3(c0.x >> 2, c0.y >> 2, c0.z), uvec4(r.x, r.y, g.x, g.y));
#elif defined(BC_OUT_EAC_R)
   /* BC4 -> EAC_R11 (or EAC_R11_SNORM), the same re-encode for one channel. */
   uvec2 r = bc_eac_encode(blk.xy, pc.format == BC_FMT_BC4_SNORM);
   imageStore(o_img, ivec3(c0.x >> 2, c0.y >> 2, c0.z), uvec4(r.x, r.y, 0u, 0u));
#elif defined(BC_OUT_ASTC)
   uvec4 o;
   if (pc.format == BC_FMT_BC7) {
      bc7_decode_block(blk);
      o = astc_from_texels();
   } else if (pc.format == BC_FMT_BC2 || pc.format == BC_FMT_BC3) {
      o = astc_from_bc23(blk, pc.format == BC_FMT_BC3);
   } else {
      o = astc_from_bc1(blk, 0, pc.format == BC_FMT_BC1_RGBA);
   }
   imageStore(o_img, ivec3(c0.x >> 2, c0.y >> 2, c0.z), o);
#elif defined(BC_OUT_ASTC7)
   /* BC7 -> ASTC 4x4. See a7_encode(). */
   bc7_decode_block(blk);
   imageStore(o_img, ivec3(c0.x >> 2, c0.y >> 2, c0.z), a7_encode());
#elif defined(BC_OUT_BC3)
   /* BC7 -> BC3: same shape, no endpoint search. See bc_bc3_encode(). */
   bc7_decode_block(blk);
   imageStore(o_img, ivec3(c0.x >> 2, c0.y >> 2, c0.z), bc_bc3_encode());
#elif defined(BC_OUT_F16) || defined(BC_OUT_RG11B10)
   if (pc.format == BC_FMT_BC6H_UF16 || pc.format == BC_FMT_BC6H_SF16) {
      bc6h_decode_store(blk, pc.format == BC_FMT_BC6H_SF16, c0, bx, by);
   } else {
      for (int t = 0; t < 16; t++)
         bc_store_texel(c0, bx, by, t, uvec3(0u));
   }
#if !defined(BC_TABLES_BUFFER)
   bc_tile_store(c0.z);
#endif
#else
   vec4 texels[16];
   float red[16], green[16];
   switch (pc.format) {
/* RGTC reaches this switch only as BC_OUT_SNORM8 (BC4/BC5_UNORM go to U32X4/EAC). Guarded out of
 * the f16/unorm8 variants to cut compile time. */
#if defined(BC_OUT_SNORM8)
   case BC_FMT_BC4_UNORM:
      bc_rgtc_block(blk, 0, false, red);
      for (int t = 0; t < 16; t++) texels[t] = vec4(red[t], 0.0, 0.0, 1.0);
      break;
   case BC_FMT_BC4_SNORM:
      bc_rgtc_block(blk, 0, true, red);
      for (int t = 0; t < 16; t++) texels[t] = vec4(red[t], 0.0, 0.0, 1.0);
      break;
   case BC_FMT_BC5_UNORM:
      bc_rgtc_block(blk, 0, false, red);
      bc_rgtc_block(blk, 64, false, green);
      for (int t = 0; t < 16; t++) texels[t] = vec4(red[t], green[t], 0.0, 1.0);
      break;
   case BC_FMT_BC5_SNORM:
      bc_rgtc_block(blk, 0, true, red);
      bc_rgtc_block(blk, 64, true, green);
      for (int t = 0; t < 16; t++) texels[t] = vec4(red[t], green[t], 0.0, 1.0);
      break;
#endif
   default: /* BC_FMT_BC7 */
/* Every variant that consumes decoded BC7 texels must be listed here: the #else writes opaque
 * black instead of failing. See also bc_store_format() in radv_meta_bc_decode.c. */
#if defined(BC_OUT_UNORM8)
      bc7_decode_block(blk);
      for (int t = 0; t < 16; t++) texels[t] = vec4(bc_px_get(t)) / 255.0;
#else
      for (int t = 0; t < 16; t++) texels[t] = vec4(0.0, 0.0, 0.0, 1.0);
#endif
      break;
   }

   /* Bounds-check every texel: the image extent need not be a multiple of 4, so the last block in
    * a row or column is partially outside it. */
   for (int t = 0; t < 16; t++) {
      int lx = t & 3, ly = t >> 2;
      if (bx * 4 + lx >= pc.extent_x || by * 4 + ly >= pc.extent_y)
         continue;
      imageStore(o_img, ivec3(c0.x + lx, c0.y + ly, c0.z), texels[t]);
   }
#endif
}
