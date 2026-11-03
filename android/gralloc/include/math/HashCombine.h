/*
 * Copyright 2021 The Android Open Source Project
 * SPDX-License-Identifier: Apache-2.0
 *
 * frameworks/native/libs/math/include/math/HashCombine.h, needed only because ui/Rect.h
 * includes it.
 */

#pragma once

#include <functional>

namespace android {

static inline void hashCombineSingleHashed(size_t& combinedHash, size_t hash) {
    combinedHash = 31 * combinedHash + hash;
}

template <typename T>
static inline void hashCombineSingle(size_t& combinedHash, const T& val) {
    hashCombineSingleHashed(combinedHash, std::hash<T>{}(val));
}

template <typename... Types>
static inline size_t hashCombine(const Types&... args) {
    size_t hash = 0;
    (hashCombineSingle(hash, args), ...);
    return hash;
}

} // namespace android
