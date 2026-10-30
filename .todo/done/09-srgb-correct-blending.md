# 09: sRGB-correct blending for UFIXED8_RGBA_SRGB

**What to build:** When the app requests `UFIXED8_RGBA_SRGB` color, fragments are converted to linear before blending and back to sRGB for output, on both paths. Today sRGB-encoded values are blended as if linear, which skews semi-transparent results. `UFIXED8_VEC4` stays linear.

**Blend convention (decided 2026-10-30, see followups/08):** wrapped-device color is premultiplied linear RGB with composited alpha, as helide and VisRTX both produce. Blend with premultiplied "over": `rgb = rgb_f + (1-a_f)*rgb_behind`, `a = a_f + (1-a_f)*a_behind`. Output the composited alpha, not 1. For sRGB, the device encodes premultiplied linear per channel. Decode, composite in linear, then re-encode.

**Blocked by:** 08 (CPU compositor supports FLOAT32_VEC4 and is UB-free)

**Status:** done

- [x] sRGB input is linearized before blending and re-encoded after, on CPU and CUDA
- [x] Opaque single-fragment pixels round-trip unchanged (within 1 LSB)
- [x] Harness test with overlapping semi-transparent geometry from two ranks compares against an analytically blended expected value
- [x] Both paths use the premultiplied convention above: CPU drops its extra `a_f*` multiply, and CUDA outputs composited alpha instead of 1
- [x] The semi-transparent test runs for `FLOAT32_VEC4` and `UFIXED8_VEC4` as well as sRGB, on CPU and CUDA, against the same analytic value
