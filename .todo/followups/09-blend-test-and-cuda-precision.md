# Follow-ups from 09 (sRGB-correct blending)

Commit reviewed: 56c8c5d. The blend work is done as the ticket asked. The
test asked for "overlapping semi-transparent geometry", and that part was
swapped for a different setup (item 1). The checkboxes:
- **sRGB linearized before blending and re-encoded after, on CPU and CUDA.**
  Met. CPU has a new `ColorFormat::RGBA8_SRGB` that decodes with a LUT on
  load and encodes on store. CUDA fragments carry the sRGB bytes exactly.
  `compositeKernel` decodes them, blends, and re-encodes. Both use the
  IEC 61966-2-1 curve, as helide does.
- **Opaque single-fragment pixels round-trip within 1 LSB.** Met by the
  1-rank `*_opaque` tests. On 2 and 3 ranks, the `*_opaque` tests check that
  rank 0's opaque background comes through unchanged over the others.
- **Harness test with overlapping semi-transparent geometry from two ranks,
  against an analytic value.** Met in substance, but not with geometry. See
  item 1.
- **Both paths premultiplied.** Met. CPU dropped the extra `a_f*`. CUDA now
  writes the composited alpha instead of 1, and both blend as
  `frag + (1-a_f)*behind`.
- **The test runs for FLOAT32_VEC4, UFIXED8_VEC4 and sRGB, on both paths.**
  Met: `{ufixed8,ufixed8-linear,float32}_{blend,opaque}` on 1 to 3 ranks,
  plus `ufixed8_blend_split`.

Nothing here blocks a queued ticket. Items 1 to 3 are choices the implementer
made alone. They change a test contract, change output, or edit vendored
code. Item 4 is a precision limit that existed before 09, which the new
tolerances now lock in. Item 5 is bookkeeping.

## 1. The blend test uses per-rank backgrounds, not semi-transparent geometry

severity: non-blocking (touches 10 and 16)

The implementer says helide forces opacity to 1 on any geometry hit, so it
cannot render semi-transparent surfaces. Instead, each rank renders a
different semi-transparent (or opaque) background. Rank 0 checks the corner
pixel, which every rank misses, against "over" of those backgrounds,
computed analytically in linear float.

What this does and doesn't cover:
- **Covered:** the blend math, sRGB decode and encode, composited alpha, and
  all three color types on both paths.
- **Not covered:** semi-transparent fragments at *different* depths, which
  have to be sorted before blending. Every background fragment has the same
  (miss) depth. So the blend order comes only from the equal-depth
  tie-break. A compositor that ignored depth and blended in rank order would
  pass. Depth sorting is still tested elsewhere, by opaque spheres only.
- **New implied contract:** "at equal depth, the lower rank is in front".
  CPU does this on purpose, with `{depth, rank}` sorting from 08. CUDA does
  it by accident: its search loops over `fromRank` ascending and uses a
  strict `<`. Nobody ever decided this. The test now depends on it.
- **README tension:** ticket 16's README tells apps to render each rank with
  a transparent background. This test depends on non-transparent
  backgrounds being blended across ranks.

**Decide:**
- Is the background-based test enough for 09? The alternative is a test
  with semi-transparent fragments at different depths. That needs a volume
  in helide, if helide's volume output is semi-transparent, or a small
  synthetic wrapped device that writes chosen color and depth buffers.
- Is "equal depth: lower rank in front" a contract? If so, ticket 16's
  README should say so. Ticket 10 should also use the same tie-break for
  nearest-depth ID/normal/albedo, if 10 implements them. If not, the test
  should not depend on it.
- Is the implementer's claim about helide's opacity correct? Check it before
  you rule out a geometry-based test.

## 2. CUDA `make_8bit` now rounds, which changes every UFIXED8 pixel on CUDA

severity: non-blocking

`make_8bit` went from `int(f*256)` (truncating, biased up) to
`f*255 + .5` (rounded, NaN-safe clamp). That matches CPU and the
`Fragment::encode8` rounding. But it changes the output by up to 1 LSB for
most UFIXED8_VEC4 and sRGB pixels on CUDA. This happens even with a single
opaque fragment. The ticket did not ask for this change. **Confirm** it.
Nothing in the queue depends on the old value.

## 3. More local edits to vendored `deepCompositing`

severity: non-blocking (adds to ticket 15's fork question)

Ticket 15 already mentions that "08/09 change the blend". The concrete 09
edits to `ext/deepCompositing/deepCompositing.{h,cu}` are:
- a new `dc::ColorFormat` enum, which replaces `bool useFloat4` in the public
  `Compositor::finish()` signature. This is an API change in the vendored
  library.
- sRGB decode/encode inside `compositeKernel`.
- composited alpha output instead of 1.
- the `make_8bit` rounding (item 2).
- the removed `(const float3&)` cast and the unused `make_rgba(float3)`.

**Add** these to 15's list when deciding whether this repo's copy becomes
the source of truth.

## 4. CUDA carries every color type in 8 bits, including FLOAT32_VEC4

severity: non-blocking

`deepCompositing.h` sets `SMALL_FRAGMENTS 1`, so `dc::Fragment` stores
color as 8-bit unorm. This was true before 09. A FLOAT32_VEC4 frame
composited on CUDA therefore has only 8-bit precision. Values outside
[0,1], such as HDR color, are clamped. The new float32 tests use a 1/255
tolerance on both paths to allow for this. That tolerance is loose for CPU,
whose float path is exact. The tests now treat 8-bit float32 on CUDA as
acceptable.

**Decide:**
- Is 8-bit precision for FLOAT32_VEC4 on CUDA acceptable? If yes, ticket
  16's README should state it. The alternative is building with full-float
  fragments for float32 frames, which costs 4x fragment traffic.
- Should the CPU float32 tolerance be tightened, so a CPU float regression
  can't hide inside 1/255?

## 5. Bookkeeping

severity: non-blocking

- Ticket 13's checkbox "The `(const float3&)fragColor` reference cast in
  `deepCompositing.cu` is gone" is now satisfied by 56c8c5d. Tick it, or
  drop it from 13.
- The checkboxes and `Status:` in `09-srgb-correct-blending.md` were not
  updated. The implementer left them to the orchestrator.
- Code review left two duplications for later: the format mapping and sRGB
  helpers are duplicated between the CPU and CUDA libraries, and the CUDA
  test format is chosen in three places. Worth tidying before the ticket 16
  port.
