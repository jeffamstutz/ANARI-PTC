# Follow-ups from 04 (Composited depth channel)

Commits reviewed: e764cb9, 65c42a6. Three of the four checkboxes are met:
- **Composited depth on both paths.** CUDA and CPU both produce composited
  depth on rank 0.
- **Mapping is refused unless enabled.** Mapping `channel.depth` without the
  app enabling it returns null, 0x0 and `ANARI_UNKNOWN`, and sends a WARNING
  through the status callback.
- **Depth is still requested internally.** The wrapped frame always gets
  FLOAT32 depth.

The fourth checkbox (harness checks on 1, 2 and 3 ranks) is met on CUDA. On
CPU, the float32 depth tests are still bypassed by ticket 08's
`PASS_REGULAR_EXPRESSION`.

One item the ticket asked for was not done (item 1). Nothing here blocks a
queued ticket.

## 1. "Depth check printed neither PASS nor FAIL" was not explained

severity: non-blocking

The ticket says: "During the investigation, the smoke test's depth check
printed neither PASS nor FAIL. Find out why as part of this ticket." The
commits, the report and the `.todo` files do not record a cause. The rewritten
`checkDepth` now prints PASS/FAIL per check, so the symptom may be gone. But
the cause is unknown.

Two candidates:
- the old CUDA gather, where a non-root rank that owned no pixels still did
  an unmatched `MPI_Send`, which could hang or abort before the print. The
  implementer fixed this, but did not link it to the symptom.
- an abort or crash in the uninitialized-depth path.

**Decide:** accept "probably the unmatched send, now fixed", or ask for a
confirmed root cause.

## 2. Vendored `ext/deepCompositing` API changed

severity: non-blocking (ticket 15 owns "ship deepCompositing?")

`dc::Compositor::finish` gained a `float *whereToWriteFinalDepth` parameter.
The changes:
- the composite kernel now writes depth;
- depth is gathered on MPI tag 1;
- `DC_NO_HIT_Z` was added;
- non-root ranks with no pixels skip their send.

This is a local fork of the vendored library. The samples
(`ext/deepCompositing/samples`, built only standalone) were already stale
(`finish(fb)`) and still are. Ticket 15 must now also decide whether these
changes go back to Ingo's upstream deepCompositing, or whether the fork is
the source of truth.

## 3. Depth is always composited and gathered

severity: non-blocking

Every frame composites and gathers depth to rank 0 on every rank, even when
the app has not enabled `channel.depth`. The cost is one float per pixel of
extra MPI traffic per frame, plus a device allocation and a D2H copy on CUDA.
The implementer chose this so all ranks follow the same MPI pattern without
having to agree on the flag. It would be easy to make this conditional later,
since every rank sees the same commits.

**Confirm** this is acceptable, or file it for ticket 13 or 16.

## 4. Depth enable latches on commit; size and color still latch on set

severity: non-blocking (ticket 05 makes size and color commit-latched)

`m_newDepthEnabled` becomes `m_depthEnabled` in `commitParameters()`.
Ticket 05 plans the same for size and color type, so this is consistent with
where 05 is headed. Until 05 lands, the frame has mixed semantics.
Unset/unset-all already clear the depth flag, which 05's "unset resets
tracked state" item should keep.

## 5. Non-FLOAT32 depth types are silently treated as "off"

severity: non-blocking

Setting `channel.depth` to a type other than `ANARI_FLOAT32`, or to a
non-`DATA_TYPE` value, disables depth without a warning. A later map does
warn that "the parameter is not set to ANARI_FLOAT32". The implementer
declined to warn at set time. Ticket 14 (diagnostics via status callback)
could pick this up.

## 6. Smoke scene changed for all tests

severity: non-blocking

Every rank but the last now renders a second, hidden sphere behind the last
rank's sphere, so color-only tests also see multi-hit pixels. This
strengthens the color check too. Noted because ticket 09 (sRGB blending) and
ticket 10 (nearest-depth ID channels) will build on this scene.

## 7. Bookkeeping

severity: non-blocking

- `04-composited-depth-channel.md` checkboxes and `Status:` were not updated.
- `FrameWrapperStatusHandler` (`std::function<void(severity, msg)>`) is the
  new route from frames to `reportMessage`. Ticket 10 (unsupported channels
  still `std::cout`) and ticket 14 should reuse it.
- On CUDA, non-root ranks also copy uninitialized `d_depth_out` to host.
  Ticket 13's "non-root ranks skip the output copy" should cover depth as
  well as color.

## Orchestrator note: the "no PASS/FAIL" diagnosis was found

The implementer's report did diagnose this; the reviewer's brief left it out. The cause was argument parsing in the original `.todo/assets/smoke.cpp`. It read arguments by position and enabled depth only when `argv[2] == "depth"`. It also silently treated an unknown `argv[1]` as ufixed8. So `smoke depth` skipped the depth block entirely.

The implementer rebuilt the old binary to confirm this:
- `smoke depth` prints no depth line.
- `smoke ufixed8 depth` prints `FAIL: depth hit=0`.

`tests/ptcSmoke.cpp` now rejects unknown arguments and always checks depth one way or the other.

## Resolution (2026-10-30, with the maintainer)

- **Item 1:** answered by the orchestrator note above.
- **Item 2:** the question of upstreaming the fork vs. treating this repo as the source of truth is added to ticket 15's deepCompositing item.
- **Item 3:** always compositing and gathering depth is accepted as is. No change.
- **Item 4:** resolved by 05.
- **Item 5:** a commit-time warning is added to ticket 14.
- **Item 6:** noted. The scene strengthens later tests.
- **Item 7:** the non-root depth copy is on ticket 13. The CPU float32 gap was closed by 08.
