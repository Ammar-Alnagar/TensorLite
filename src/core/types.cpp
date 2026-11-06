#include "tensorlite/core/types.hpp"
#include <cmath>
#include <cstring>
#include <algorithm>

namespace tensorlite {

fp8_e4m3 fp8_e4m3::from_float(float f) {
    // Bit representation of IEEE 754 float32:
    // 1 bit sign, 8 bits exponent (bias 127), 23 bits mantissa
    uint32_t u;
    std::memcpy(&u, &f, sizeof(float));

    uint32_t sign = (u >> 31) & 0x1;
    int32_t exp = ((u >> 23) & 0xFF) - 127;
    uint32_t mant = u & 0x7FFFFF;

    if (std::isnan(f)) {
        return fp8_e4m3(static_cast<uint8_t>((sign << 7) | 0x7F)); // NaN
    }

    if (std::isinf(f)) {
        // Clamp to max finite or NaN. OCP E4M3: max finite is 448 (exp=15, mant=6: 0b01111110)
        return fp8_e4m3(static_cast<uint8_t>((sign << 7) | 0x7E));
    }

    if (f == 0.0f) {
        return fp8_e4m3(static_cast<uint8_t>(sign << 7));
    }

    // Target E4M3 has bias 7:
    // exponent in [1..15], mantissa 3 bits.
    int32_t target_exp = exp + 7;

    if (target_exp >= 15) {
        // Overflow -> saturate to max finite (448.0)
        return fp8_e4m3(static_cast<uint8_t>((sign << 7) | 0x7E));
    } else if (target_exp <= 0) {
        // Subnormal or zero in FP8 E4M3
        // Shift mantissa
        int shift = 1 - target_exp;
        if (shift > 4) {
            return fp8_e4m3(static_cast<uint8_t>(sign << 7));
        }
        uint32_t full_mant = (1 << 3) | (mant >> 20); // 3 bits mantissa + hidden bit
        uint32_t sub_mant = full_mant >> shift;
        return fp8_e4m3(static_cast<uint8_t>((sign << 7) | (sub_mant & 0x07)));
    } else {
        // Normal number: take top 3 bits of 23-bit mantissa with rounding
        uint32_t mant3 = (mant >> 20) & 0x7;
        // Check round bit
        if ((mant >> 19) & 0x1) {
            mant3++;
            if (mant3 > 0x7) {
                target_exp++;
                mant3 = 0;
                if (target_exp >= 15) {
                    return fp8_e4m3(static_cast<uint8_t>((sign << 7) | 0x7E));
                }
            }
        }
        return fp8_e4m3(static_cast<uint8_t>((sign << 7) | ((target_exp & 0xF) << 3) | (mant3 & 0x7)));
    }
}

float fp8_e4m3::to_float() const {
    uint32_t sign = (bits >> 7) & 0x1;
    uint32_t exp = (bits >> 3) & 0xF;
    uint32_t mant = bits & 0x7;

    if (exp == 0xF && mant == 0x7) {
        return std::numeric_limits<float>::quiet_NaN();
    }

    if (exp == 0) {
        if (mant == 0) {
            return sign ? -0.0f : 0.0f;
        }
        // Subnormal: (-1)^sign * 2^(-6) * (mant / 8)
        float val = std::ldexp(static_cast<float>(mant) / 8.0f, -6);
        return sign ? -val : val;
    }

    // Normal: (-1)^sign * 2^(exp - 7) * (1 + mant / 8)
    int real_exp = static_cast<int>(exp) - 7;
    float val = std::ldexp(1.0f + static_cast<float>(mant) / 8.0f, real_exp);
    return sign ? -val : val;
}

} // namespace tensorlite
