/*
 * This file is auto-generated.  DO NOT MODIFY.
 * Using: C:\\Users\\alpha\\AppData\\Local\\Android\\Sdk\\build-tools\\36.0.0\\aidl.exe --lang=ndk --structured --stability=vintf --min_sdk_version=34 -I hw-interfaces\\graphics\\common\\aidl\\aidl_api\\android.hardware.graphics.common\\5 -o gen-aidl\\src -h gen-aidl\\include hw-interfaces\\graphics\\common\\aidl\\aidl_api\\android.hardware.graphics.common\\5\\android\\hardware\\graphics\\common\\Rect.aidl
 *
 * DO NOT CHECK THIS FILE INTO A CODE TREE (e.g. git, etc..).
 * ALWAYS GENERATE THIS FILE FROM UPDATED AIDL COMPILER
 * AS A BUILD INTERMEDIATE ONLY. THIS IS NOT SOURCE CODE.
 */
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>
#include <android/binder_interface_utils.h>
#include <android/binder_parcelable_utils.h>
#include <android/binder_to_string.h>
#ifdef BINDER_STABILITY_SUPPORT
#include <android/binder_stability.h>
#endif  // BINDER_STABILITY_SUPPORT

namespace aidl {
namespace android {
namespace hardware {
namespace graphics {
namespace common {
class Rect {
public:
  typedef std::false_type fixed_size;
  static const char* descriptor;

  int32_t left = 0;
  int32_t top = 0;
  int32_t right = 0;
  int32_t bottom = 0;

  binder_status_t readFromParcel(const AParcel* parcel);
  binder_status_t writeToParcel(AParcel* parcel) const;

  inline bool operator==(const Rect& _rhs) const {
    return std::tie(left, top, right, bottom) == std::tie(_rhs.left, _rhs.top, _rhs.right, _rhs.bottom);
  }
  inline bool operator<(const Rect& _rhs) const {
    return std::tie(left, top, right, bottom) < std::tie(_rhs.left, _rhs.top, _rhs.right, _rhs.bottom);
  }
  inline bool operator!=(const Rect& _rhs) const {
    return !(*this == _rhs);
  }
  inline bool operator>(const Rect& _rhs) const {
    return _rhs < *this;
  }
  inline bool operator>=(const Rect& _rhs) const {
    return !(*this < _rhs);
  }
  inline bool operator<=(const Rect& _rhs) const {
    return !(_rhs < *this);
  }

  static const ::ndk::parcelable_stability_t _aidl_stability = ::ndk::STABILITY_VINTF;
  inline std::string toString() const {
    std::ostringstream _aidl_os;
    _aidl_os << "Rect{";
    _aidl_os << "left: " << ::android::internal::ToString(left);
    _aidl_os << ", top: " << ::android::internal::ToString(top);
    _aidl_os << ", right: " << ::android::internal::ToString(right);
    _aidl_os << ", bottom: " << ::android::internal::ToString(bottom);
    _aidl_os << "}";
    return _aidl_os.str();
  }
};
}  // namespace common
}  // namespace graphics
}  // namespace hardware
}  // namespace android
}  // namespace aidl
