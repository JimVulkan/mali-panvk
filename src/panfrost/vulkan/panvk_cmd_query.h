/*
 * Copyright © 2024 Collabora Ltd.
 * SPDX-License-Identifier: MIT
 */

#ifndef PANVK_CMD_QUERY_H
#define PANVK_CMD_QUERY_H

#ifndef PAN_ARCH
#error "PAN_ARCH must be defined"
#endif

#include "genxml/gen_macros.h"

struct panvk_occlusion_query_state {
#if PAN_ARCH >= 10
   uint64_t syncobj;
#endif
   uint64_t ptr;
   enum mali_occlusion_mode mode;
};

/* VK_QUERY_TYPE_PIPELINE_STATISTICS. Mali has no pipeline counters, so the counts are derived
 * from the parameters of direct draws and dispatches on the CPU (as gallium panfrost does for
 * primitives generated) and written at the query end, one value per bit of
 * VkQueryPipelineStatisticFlagBits. Indirect draws/dispatches and fragment invocations are not
 * counted; geometry and tessellation counts assume one output primitive per input. */
#define PANVK_PIPELINE_STAT_COUNT 11

struct panvk_pipeline_stats_state {
   bool active;
   uint64_t counts[PANVK_PIPELINE_STAT_COUNT];
};

#if PAN_ARCH >= 10
struct panvk_prims_generated_query_state {
   uint64_t syncobj;
   uint64_t ptr;
};
#endif

#endif
