/*
 * This file is auto-generated.  DO NOT MODIFY.
 *
 * Hand-synthesized to match the platform aidl output for
 * android.hardware.graphics.common.BufferUsage (frozen v5). The Android SDK
 * build-tools aidl rejects VENDOR_MASK_HI ((1L*0xffff)<<48) as a signed-long
 * "constant expression overflow" -- a pedantry the platform aidl compiler does
 * not apply. Every value below is the exact frozen ABI value; the two high
 * masks are written as the same bit patterns via explicit int64_t casts so a
 * standards-conforming compiler accepts them. Byte-for-byte enum-value
 * identical to a platform build.
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
enum class BufferUsage : int64_t {
  CPU_READ_MASK = 0xfL,
  CPU_READ_NEVER = 0L,
  CPU_READ_RARELY = 2L,
  CPU_READ_OFTEN = 3L,
  CPU_WRITE_MASK = 240L,
  CPU_WRITE_NEVER = 0L,
  CPU_WRITE_RARELY = 32L,
  CPU_WRITE_OFTEN = 48L,
  GPU_TEXTURE = 256L,
  GPU_RENDER_TARGET = 512L,
  COMPOSER_OVERLAY = 2048L,
  COMPOSER_CLIENT_TARGET = 4096L,
  PROTECTED = 16384L,
  COMPOSER_CURSOR = 32768L,
  VIDEO_ENCODER = 65536L,
  CAMERA_OUTPUT = 131072L,
  CAMERA_INPUT = 262144L,
  RENDERSCRIPT = 1048576L,
  VIDEO_DECODER = 4194304L,
  SENSOR_DIRECT_DATA = 8388608L,
  GPU_DATA_BUFFER = 16777216L,
  GPU_CUBE_MAP = 33554432L,
  GPU_MIPMAP_COMPLETE = 67108864L,
  HW_IMAGE_ENCODER = 134217728L,
  FRONT_BUFFER = 4294967296L,
  VENDOR_MASK = 4026531840L,
  VENDOR_MASK_HI = static_cast<int64_t>(0xffff000000000000ULL) /* -281474976710656 */,
};

}  // namespace common
}  // namespace graphics
}  // namespace hardware
}  // namespace android
}  // namespace aidl
namespace ndk {
namespace internal {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wswitch-enum"
template <>
constexpr inline std::array<aidl::android::hardware::graphics::common::BufferUsage, 27> enum_values<aidl::android::hardware::graphics::common::BufferUsage> = {
  aidl::android::hardware::graphics::common::BufferUsage::CPU_READ_MASK,
  aidl::android::hardware::graphics::common::BufferUsage::CPU_READ_NEVER,
  aidl::android::hardware::graphics::common::BufferUsage::CPU_READ_RARELY,
  aidl::android::hardware::graphics::common::BufferUsage::CPU_READ_OFTEN,
  aidl::android::hardware::graphics::common::BufferUsage::CPU_WRITE_MASK,
  aidl::android::hardware::graphics::common::BufferUsage::CPU_WRITE_NEVER,
  aidl::android::hardware::graphics::common::BufferUsage::CPU_WRITE_RARELY,
  aidl::android::hardware::graphics::common::BufferUsage::CPU_WRITE_OFTEN,
  aidl::android::hardware::graphics::common::BufferUsage::GPU_TEXTURE,
  aidl::android::hardware::graphics::common::BufferUsage::GPU_RENDER_TARGET,
  aidl::android::hardware::graphics::common::BufferUsage::COMPOSER_OVERLAY,
  aidl::android::hardware::graphics::common::BufferUsage::COMPOSER_CLIENT_TARGET,
  aidl::android::hardware::graphics::common::BufferUsage::PROTECTED,
  aidl::android::hardware::graphics::common::BufferUsage::COMPOSER_CURSOR,
  aidl::android::hardware::graphics::common::BufferUsage::VIDEO_ENCODER,
  aidl::android::hardware::graphics::common::BufferUsage::CAMERA_OUTPUT,
  aidl::android::hardware::graphics::common::BufferUsage::CAMERA_INPUT,
  aidl::android::hardware::graphics::common::BufferUsage::RENDERSCRIPT,
  aidl::android::hardware::graphics::common::BufferUsage::VIDEO_DECODER,
  aidl::android::hardware::graphics::common::BufferUsage::SENSOR_DIRECT_DATA,
  aidl::android::hardware::graphics::common::BufferUsage::GPU_DATA_BUFFER,
  aidl::android::hardware::graphics::common::BufferUsage::GPU_CUBE_MAP,
  aidl::android::hardware::graphics::common::BufferUsage::GPU_MIPMAP_COMPLETE,
  aidl::android::hardware::graphics::common::BufferUsage::HW_IMAGE_ENCODER,
  aidl::android::hardware::graphics::common::BufferUsage::FRONT_BUFFER,
  aidl::android::hardware::graphics::common::BufferUsage::VENDOR_MASK,
  aidl::android::hardware::graphics::common::BufferUsage::VENDOR_MASK_HI,
};
#pragma clang diagnostic pop
}  // namespace internal
}  // namespace ndk
