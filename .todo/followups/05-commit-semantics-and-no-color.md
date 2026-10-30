# Follow-ups from 05 (Frame parameter tracking)

Commits reviewed: 157bf31, afb8f83. All five checkboxes are met:
- **Commit semantics.** Set, unset and unset-all write pending state for
  size, color type and depth. `anariCommitParameters` copies it to committed
  state. Buffers are reallocated at render time from the committed state.
- **Unset resets tracking.** Unset-all resets everything and still asks the
  wrapped frame for FLOAT32 depth.
- **Color-type-only change reallocates.** One base-class `updateBuffers()`
  compares size and color type on both paths.
- **No `channel.color`.** Rendering without it no longer crashes. Mapping
  color returns null, 0x0 and `ANARI_UNKNOWN`, with a WARNING.
- **Harness.** Covers resize, retype, render without color, unset color,
  unset-all and uncommitted changes, on 1, 2 and 3 ranks.

Nothing here blocks a queued ticket. Items 1 and 2 are bookkeeping that later
tickets need. The rest are choices for you to confirm.

## 1. Ticket 13's "one `prepare()` per frame" is already done

severity: non-blocking (bookkeeping for ticket 13)

The first `dc::DeviceInterface dcDev = m_deepComp.prepare();` in
`CUDAFrameWrapper::composite()` was removed, and its result was never used.
The only call left is the one passed to the composite kernel launch
(`ptc/CUDAFrameWrapper.cu:141`). So ticket 13's second checkbox is met, but
its text still lists the item as open. Tick it, or tell 13 to verify it and
skip it. Ticket 13 still needs to add its own check that there is one
`prepare()` per frame.

## 2. CPU retype and float32 coverage is only nominal until ticket 08

severity: non-blocking (ticket 08 inherits this)

On CPU, every test whose arguments include `float32` passes by matching the
ticket-08 "FLOAT32_VEC4 not implemented" regex. That includes:
- `retype_float32_to_ufixed8`;
- `retype_ufixed8_to_float32`;
- `float32_resize`;
- `float32_uncommitted`.

So on CPU, the only real color-type reallocation test is
`retype_ufixed8_linear_to_ufixed8`, where both types are 4 bytes per pixel.
When ticket 08 removes the expected-fail, these tests become real CPU tests
of 05's behavior for the first time. If they fail then, the cause may be in
05's code, not 08's. Ticket 08's "harness float test no longer expected-fail"
checkbox should name these tests too.

## 3. The ticket's CPU color-type bug did not reproduce

severity: non-blocking

The ticket says "The CPU path ignores a color-type change when the size is
unchanged." The implementer found that HEAD already compared both size and
color type on CPU. The fix is structural: one shared check replaces two
per-path copies. Either the ticket text was stale, or the bug was somewhere
else (for example in compositor state that was not reset). **Confirm** the
ticket text was stale.

## 4. A frame without `channel.color` still composites depth

severity: non-blocking

With no color type, the compositor runs on transparent-black input:
- a zero buffer on CPU;
- a `cudaMemset` buffer on CUDA.

So depth still composites. The color buffer is allocated at 4 bytes per
pixel but cannot be mapped. The other choice was to skip compositing
entirely when color is missing. That would leave depth unmappable, or would
need a depth-only path. **Confirm** that depth-only frames should work.

## 5. A wrong-typed `size` or `channel.color` counts as unset

severity: non-blocking

Setting `size` with a type other than `UINT32_VEC2`, or `channel.color` with
a type other than `DATA_TYPE`, now resets the tracked value to its default.
This matches helium's `getParam`, which the wrapped frame uses. The value is
still forwarded to the wrapped frame. There is no warning. Ticket 14 could
add one, which is the same gap as 04's follow-up item 5 for
`channel.depth`.

A related gap is out of scope here. `colorPixelSize()` treats every color
type other than `FLOAT32_VEC4` as 4 bytes. That includes types the
compositors cannot handle, such as `FLOAT32` and `UFIXED8_VEC3`. This
behavior predates 05.

## 6. Map permission and size latch at render time, not at commit

severity: non-blocking

afb8f83 changed when the depth map permission latches. 04 latched it on
commit. Now it latches when the frame renders (`m_renderedDepthEnabled`), as
size and color type do. So a commit with no render after it does not change
what `anariMapFrame` reports. This changes 04's behavior, and it matches
ANARI's rule that a map shows the last render.

## 7. New WARNING on color map overlaps ticket 14

severity: non-blocking

Mapping color when the frame has no color now sends a WARNING through
`m_reportStatus`. This follows 04's pattern for depth. Ticket 14 should treat
it as already routed. The `std::cout` for unsupported channels is unchanged
and stays with ticket 10.

## 8. Bookkeeping

severity: non-blocking

- `05-frame-parameter-tracking.md` checkboxes and `Status:` were not
  updated. 04 had the same gap.
- Every two-frame test also flips `channel.depth` between frames. This gives
  more coverage, but a failure does not show whether the depth flip or the
  size or type change caused it.
- On CPU, if `allocate()` throws for float32, `updateBuffers()` now frees all
  buffers before rethrowing. The exception still escapes `anariRenderFrame`.
  Ticket 08 owns that.

## Resolution (2026-10-30, with the maintainer)

- **Item 1:** noted on ticket 13.
- **Item 2:** resolved by 08. Those CPU float32 tests now run for real and pass.
- **Items 3, 4 and 6:** confirmed as landed. The ticket text was stale, depth-only frames are supported, and map state is fixed at render.
- **Item 5:** a commit-time wrong-type warning is added to ticket 14. 08's `isCompositableColorType` covers the `colorPixelSize` aside.
- **Item 7:** informational.
- **Item 8:** 08 fixed the exception escape.
