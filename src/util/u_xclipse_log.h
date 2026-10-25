/*
 * Copyright 2026 JimVulkan
 * SPDX-License-Identifier: MIT
 */

/* The Xclipse log switch for code outside src/amd: RADV_XCLIPSE_LOG (env) or
 * debug.radv_xclipse_log (property), level 1 or more. Off by default. Same switch as
 * ac_xclipse_log.h; read on every call, so use it only for once-per-process lines. */
#ifndef U_XCLIPSE_LOG_H
#define U_XCLIPSE_LOG_H

#include <stdbool.h>
#include <stdlib.h>

#ifdef __ANDROID__
#include <sys/system_properties.h>
#endif

static inline bool
u_xclipse_log_on(void)
{
   const char *e = getenv("RADV_XCLIPSE_LOG");
   if (e && e[0])
      return atoi(e) >= 1;
#ifdef __ANDROID__
   char v[PROP_VALUE_MAX] = {0};
   return __system_property_get("debug.radv_xclipse_log", v) > 0 && atoi(v) >= 1;
#else
   return false;
#endif
}

#endif /* U_XCLIPSE_LOG_H */
