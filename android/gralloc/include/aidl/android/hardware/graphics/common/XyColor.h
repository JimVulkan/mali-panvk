/*
 * This file is auto-generated.  DO NOT MODIFY.
 * Using: C:\\Users\\alpha\\AppData\\Local\\Android\\Sdk\\build-tools\\36.0.0\\aidl.exe --lang=ndk --structured --stability=vintf --min_sdk_version=34 -I hw-interfaces\\graphics\\common\\aidl\\aidl_api\\android.hardware.graphics.common\\5 -o gen-aidl\\src -h gen-aidl\\include hw-interfaces\\graphics\\common\\aidl\\aidl_api\\android.hardware.graphics.common\\5\\android\\hardware\\graphics\\common\\XyColor.aidl
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
class XyColor {
public:
  typedef std::false_type fixed_size;
  static const char* descriptor;

  float x = 0.000000f;
  float y = 0.000000f;

  binder_status_t readFromParcel(const AParcel* parcel);
  binder_status_t writeToParcel(AParcel* parcel) const;

  inline bool operator==(const XyColor& _rhs) const {
    return std::tie(x, y) == std::tie(_rhs.x, _rhs.y);
  }
  inline bool operator<(const XyColor& _rhs) const {
    return std::tie(x, y) < std::tie(_rhs.x, _rhs.y);
  }
  inline bool operator!=(const XyColor& _rhs) const {
    return !(*this == _rhs);
  }
  inline bool operator>(const XyColor& _rhs) const {
    return _rhs < *this;
  }
  inline bool operator>=(const XyColor& _rhs) const {
    return !(*this < _rhs);
  }
  inline bool operator<=(const XyColor& _rhs) const {
    return !(_rhs < *this);
  }

  static const ::ndk::parcelable_stability_t _aidl_stability = ::ndk::STABILITY_VINTF;
  inline std::string toString() const {
    std::ostringstream _aidl_os;
    _aidl_os << "XyColor{";
    _aidl_os << "x: " << ::android::internal::ToString(x);
    _aidl_os << ", y: " << ::android::internal::ToString(y);
    _aidl_os << "}";
    return _aidl_os.str();
  }
};
}  // namespace common
}  // namespace graphics
}  // namespace hardware
}  // namespace android
}  // namespace aidl
