# 09: sRGB-correct blending for UFIXED8_RGBA_SRGB

**What to build:** When the app requests `UFIXED8_RGBA_SRGB` color, fragments are converted to linear before blending and back to sRGB for output, on both paths. Today sRGB-encoded values are blended as if linear, which skews semi-transparent results. `UFIXED8_VEC4` stays linear.

**Blocked by:** 08 (CPU compositor supports FLOAT32_VEC4 and is UB-free)

**Status:** ready-for-agent

- [ ] sRGB input is linearized before blending and re-encoded after, on CPU and CUDA
- [ ] Opaque single-fragment pixels round-trip unchanged (within 1 LSB)
- [ ] Harness test with overlapping semi-transparent geometry from two ranks compares against an analytically blended expected value
