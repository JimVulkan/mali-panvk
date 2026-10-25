/*
 * This file is auto-generated.  DO NOT MODIFY.
 * Using: C:\\Users\\alpha\\AppData\\Local\\Android\\Sdk\\build-tools\\36.0.0\\aidl.exe --lang=ndk --structured --stability=vintf --min_sdk_version=34 -I hw-interfaces\\graphics\\common\\aidl\\aidl_api\\android.hardware.graphics.common\\5 -o gen-aidl\\src -h gen-aidl\\include hw-interfaces\\graphics\\common\\aidl\\aidl_api\\android.hardware.graphics.common\\5\\android\\hardware\\graphics\\common\\Interlaced.aidl
 *
 * DO NOT CHECK THIS FILE INTO A CODE TREE (e.g. git, etc..).
 * ALWAYS GENERATE THIS FILE FROM UPDATED AIDL COMPILER
 * AS A BUILD INTERMEDIATE ONLY. THIS IS NOT SOURCE CODE.
 */
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <android/binder_enums.h>
#ifdef BINDER_STABILITY_SUPPORT
#include <android/binder_stability.h>
#endif  // BINDER_STABILITY_SUPPORT

namespace aidl {
namespace android {
namespace hardware {
namespace graphics {
namespace common {
enum class Interlaced : int64_t {
  NONE = 0L,
  TOP_BOTTOM = 1L,
  RIGHT_LEFT = 2L,
};

}  // namespace common
}  // namespace graphics
}  // namespace hardware
}  // namespace android
}  // namespace aidl
namespace aidl {
namespace android {
namespace hardware {
namespace graphics {
namespace common {
[[nodiscard]] static inline std::string toString(Interlaced val) {
  switch(val) {
  case Interlaced::NONE:
    return "NONE";
  case Interlaced::TOP_BOTTOM:
    return "TOP_BOTTOM";
  case Interlaced::RIGHT_LEFT:
    return "RIGHT_LEFT";
  default:
    return std::to_string(static_cast<int64_t>(val));
  }
}
}  // namespace common
}  // namespace graphics
}  // namespace hardware
}  // namespace android
}  // namespace aidl
namespace ndk {
namespace internal {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wc++17-extensions"
template <>
constexpr inline std::array<aidl::android::hardware::graphics::common::Interlaced, 3> enum_values<aidl::android::hardware::graphics::common::Interlaced> = {
  aidl::android::hardware::graphics::common::Interlaced::NONE,
  aidl::android::hardware::graphics::common::Interlaced::TOP_BOTTOM,
  aidl::android::hardware::graphics::common::Interlaced::RIGHT_LEFT,
};
#pragma clang diagnostic pop
}  // namespace internal
}  // namespace ndk
