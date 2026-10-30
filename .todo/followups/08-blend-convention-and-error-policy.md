# Follow-ups from 08 (CPU compositor FLOAT32_VEC4, UB-free)

Commits reviewed: 0ded9fd, 55d87fc. All four checkboxes are met, with the
caveats below:
- **FLOAT32_VEC4 on 1, 2 and 3 ranks.** The CPU float32 tests now run for
  real: plain, depth, resize, retype both ways, and uncommitted. But they
  only threshold-check opaque sphere centers (`fabs(hit - want) < 0.05`).
  Semi-transparent blending in float32 is untested (item 4).
- **No UB.** The end-iterator deref is gone, because fragments now sort as
  `{depth, rank}` pairs over vector iterators. Type punning goes through
  memcpy, as it already did after 04. Float-to-unorm8 now clamps NaN and
  out-of-range values. The ASan/UBSan run is clean, but the old code was
  clean too, so the run proves nothing here. The fix is verified by
  inspection only.
- **Errors via the status callback.** Every `PTCDevice` and `PTCLibrary`
  entry point catches exceptions and reports them as ERROR. That includes
  the `bit_cast` size check.
- **Expected-fail removed.** The "not implemented" `PASS_REGULAR_EXPRESSION`
  is gone. This also settles 05 follow-up item 2.

One item blocks ticket 09 (item 1). The rest are choices to confirm, or notes
for later tickets.

## 1. CPU and CUDA use different blend conventions, not only different output alpha

severity: blocking (ticket 09)

The implementer described the divergence as "CPU keeps composited alpha,
CUDA forces alpha to 1". The real difference is larger:

- **CPU** (`cpu_compositing/CPUCompositor.cpp`, `compositeMyRange`) treats
  colors as **straight** (non-premultiplied):
  - `rgb = a_f*rgb_f + (1-a_f)*rgb_behind`
  - `a = a_f + (1-a_f)*a_behind`
- **CUDA** (`ext/deepCompositing/deepCompositing.cu:460-475`) treats colors
  as **premultiplied**. It accumulates front to back with
  `color += (1-alpha) * fragColor.rgb`. The `* fragColor.w` factor is
  commented out. It then writes alpha = 1.

For any semi-transparent fragment, the two paths give different RGB, not
just different alpha. Ticket 16 says the README will document
"non-premultiplied over". That matches CPU, not CUDA.

Ticket 09 has to linearize sRGB before blending and test both paths against
one analytically blended value. It cannot do that until one convention is
chosen. sRGB also interacts with premultiplication: you cannot linearize
premultiplied sRGB values per channel and get the right answer. Ticket 09
does not mention this today. Its implementer would have to pick a convention
alone, and would likely rewrite the CUDA kernel's math to do it.

**Decide before 09 starts:**
- **Input convention.** Is the wrapped device's color straight or
  premultiplied? Check what helide (or the ANARI spec) writes.
- **Output alpha.** Should it be the composited alpha, or forced to 1?
- **Ownership.** Should ticket 09 bring CUDA (or CPU) in line? If so, add a
  checkbox to 09 so the fix doesn't happen silently inside the sRGB work.

## 2. Unsupported color types now fail at render time

severity: non-blocking (overlaps ticket 14, and 05 follow-up item 5)

`FrameWrapper::updateBuffers()` now throws, and so reports an ERROR, for any
`channel.color` type other than UFIXED8_VEC4, UFIXED8_RGBA_SRGB or
FLOAT32_VEC4. This applies on both paths. Before, FLOAT32_VEC3,
UFIXED8_VEC3 and the rest were composited as garbage RGBA8. The code
reviewer called it scope creep.

This is new, user-visible behavior that the ticket did not ask for. Choices
to confirm:
- **Reject at all?** The alternative is to support or convert more types.
- **When to reject.** It currently happens at render. Rejecting at commit or
  set would also be possible. The wrapped device may accept the type, so the
  wrapped frame still renders. Only ptc's compositing is refused.
- **Severity and wording.** It is reported as ERROR with
  `ANARI_STATUS_UNKNOWN_ERROR`. Ticket 14 may want a more specific status
  code, such as `INVALID_ARGUMENT`.

After the failed render, mapping color sends a WARNING saying the frame was
rendered without `channel.color`. That is misleading, because color was set
with an unsupported type.

## 3. The blanket `guarded()` error policy

severity: non-blocking

Every `PTCDevice` entry point now runs inside `guarded()`. It catches any
exception, reports `"<anariEntryPoint>: <what()>"` as ERROR through the
status callback, and returns null or 0. `PTCLibrary::newDevice` and
`getDeviceExtensions` do the same. This is reasonable and matches ANARI's
no-exceptions rule. But it sets policy that tickets 14 and 16 will inherit.
**Confirm:**
- **Status code and severity.** It always uses ERROR with
  `ANARI_STATUS_UNKNOWN_ERROR`. FATAL_ERROR is never used, not even for a
  failed `initDevice()`, such as when the wrapped library cannot load.
- **Asymmetric failures hang.** If only one rank throws in `allocate()` or
  `composite()` (OOM, a CUDA error), that rank returns and the others hang
  in collectives. A comment in `updateBuffers()` now says this. There is no
  plan for it: an abort, an agreement collective, or an accepted limit. It
  could become a line in ticket 11 or ticket 16's README.
- **What a failed render leaves behind.**
  - The frame completion callback does not fire. This extends 06 follow-up
    item 2.
  - `frameReady` returns normally.
  - Maps are refused, with the misleading warning from item 2.

  Decide whether the callback should fire on error, and whether
  `frameReady` should report the error.

## 4. Coverage gaps

severity: non-blocking

- **Float32 blending is untested.** The float32 tests check opaque centers
  with a 0.05 tolerance only, so a float32 blend bug would pass. Ticket 09's
  semi-transparent test would cover this if it runs on float32 as well as
  sRGB. Consider saying so in 09.
- **No test exercises the UB fix.** Nothing tests negative or NaN depth
  ordering, or the unorm8 clamp.

## 5. Notes for later tickets

severity: non-blocking

- **Ticket 13: the CUDA `(const float3&)fragColor` cast** in
  `deepCompositing.cu:465` is still there. The implementer left it for 13,
  but 13's text does not list it. Add it, or let 09 drop it while it
  rewrites that blend (see item 1).
- **MPI counts are `int`.** With float32 colors, the element counts
  overflow 4x sooner, at about 536M pixels per rank exchange. This is
  unlikely to matter, but worth noting for 16.
- **Style.** The color-format switch is duplicated. `p2p` uses parallel
  count/offset vectors. The rethrow/report code is duplicated between
  `PTCDevice` and `PTCLibrary`. Worth tidying before the ticket-16 port.
- **Bookkeeping.** The checkboxes and `Status:` in
  `08-cpu-compositor-float-and-ub.md` were not updated. 03 to 06 had the
  same gap.
