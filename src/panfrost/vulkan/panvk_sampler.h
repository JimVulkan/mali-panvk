/*
 * Copyright © 2021 Collabora Ltd.
 * SPDX-License-Identifier: MIT
 */

#ifndef PANVK_SAMPLER_H
#define PANVK_SAMPLER_H

#ifndef PAN_ARCH
#error "PAN_ARCH must be defined"
#endif

#include <stdint.h>

#include "vk_sampler.h"

/* We use 2 sampler planes for YCbCr conversion with different filters for
 * the Y and CbCr components.
 */
#define PANVK_MAX_DESCS_PER_SAMPLER 2

struct panvk_sampler {
   struct vk_sampler vk;
   struct mali_sampler_packed descs[PANVK_MAX_DESCS_PER_SAMPLER];
   uint8_t desc_count;
};

VK_DEFINE_NONDISP_HANDLE_CASTS(panvk_sampler, vk.base, VkSampler,
                               VK_OBJECT_TYPE_SAMPLER)

/* Packs the first sampler plane of a sampler state, for samplers that have
 * no VkSampler object (descriptor heap writes, embedded samplers). */
void panvk_per_arch(sampler_state_fill_desc)(
   const struct vk_sampler_state *state, struct mali_sampler_packed *desc);

#endif
