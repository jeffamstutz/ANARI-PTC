# Follow-ups from 06 (Frame completion callback)

Commits reviewed: 231e736, 781a1f8. All five checkboxes are met:
- **Intercepted.** `frameCompletionCallback` and `frameCompletionCallbackUserData`
  are no longer forwarded to the wrapped frame. Both are tracked as pending
  and committed state, with 05's set/unset/unset-all semantics.
- **After compositing, once per render.** `FrameWrapper::renderFrame()` calls
  the callback after `composite()`. The CUDA path's output copy is a blocking
  `cudaMemcpy`, so the image is ready when the callback runs.
- **App handles and user data.** The callback receives the app's user data,
  the ptc device handle and the ptc frame handle.
- **Unset stops calls.** Covered for `anariUnsetParameter` and
  `anariUnsetAllParameters`.
- **Harness.** `--callback` checks the handles, the user data and the call
  count. Rank 0 maps the composited color inside the callback. The tests
  `callback`, `callback_resize` and `callback_unset_all` run on 1, 2 and 3
  ranks.

Nothing here blocks a queued ticket. Items 1 to 3 are choices the implementer
made alone, and they set the semantics that ticket 16's README will document.
The rest are notes for later tickets.

## 1. The callback runs synchronously, inside `anariRenderFrame`

severity: non-blocking

ptc renders synchronously, so the callback runs on the app's thread before
`anariRenderFrame` returns. The extension encourages calling it from a
background thread instead. The current choice fits ptc's synchronous render
model. Moving to a background thread would need an async render/composite
model, which is a larger design change.

There is no guard against re-entrant calls. For example, the app could call
`anariRenderFrame` or `anariCommitParameters` on the same frame from inside
the callback. helide refuses that case, but ptc would recurse into
`renderFrame()`. Under MPI this happens on every rank, so it would be
collective.

**Decide:**
- whether synchronous is the intended contract, so that ticket 16 documents
  it;
- whether re-entrant use should be refused with a status message.

The implementer suggested ticket 14 for this. But 14 is about routing
diagnostics, so at most the refusal message belongs there. The guard itself
needs its own home, or a line added to 14.

## 2. The callback fires on every rank

severity: non-blocking

Every rank's callback fires, not only rank 0's. Only rank 0 holds the
composited image. Firing everywhere keeps the ranks symmetric: every rank
called `anariRenderFrame`, and an app may do collective work in the
callback. The alternative is to fire on rank 0 only. **Confirm** that firing
on every rank is the intended semantics, before ticket 16's README documents
it.

Two edge cases come with this choice:
- The callback fires for 0x0 frames, where compositing is skipped.
- It does not fire if `composite()` throws. Exceptions escaping
  `anariRenderFrame` are ticket 08's concern.

If ticket 07 (parked) adds per-frame communicators, "every rank" should
become "every rank in the frame's communicator".

## 3. `KHR_FRAME_COMPLETION_CALLBACK` is advertised only if the wrapped device advertises it

severity: non-blocking (relates to ticket 12)

ptc now implements the extension itself and never forwards the callback. But
its `extension` list still comes from the wrapped device's list, plus
`KHR_DATA_PARALLEL_MPI`. So if the wrapped device lacks the extension, ptc
honors the callback without advertising it.

This matters once ticket 12 lets the app choose other wrapped libraries.
**Decide** whether ptc should always advertise the extension. The same
question applies to any extension ptc implements itself instead of
forwarding.

## 4. The frame does not retain the ptc device

severity: non-blocking (ticket 11)

`FrameWrapper` stores the ptc device handle without retaining it. If the
device is released before the frame and the frame then renders, the callback
receives a stale device handle. This adds a case to ticket 11's lifetime
work, which already covers device release ordering.

## 5. CUDA current device inside the callback

severity: non-blocking (ticket 13)

The callback runs on the app thread right after compositing. So any change
to the thread's current CUDA device that ptc has not restored is visible to
the app's callback code. Ticket 13's first checkbox covers this. Its test
could also check the current device from inside the callback.

## 6. Coverage and bookkeeping

severity: non-blocking

- Unsetting only `frameCompletionCallbackUserData` is untested. Unsetting it
  should pass null to a still-set callback.
- A `frameCompletionCallbackUserData` of type `ANARI_VOID_POINTER` is read as
  the pointer value in `mem`. This matches ANARI's convention for pointer
  parameters. Any other type is treated as an unset, with no warning, as 05
  did for `size` and `channel.color`.
- The checkboxes and `Status:` in `06-frame-completion-callback.md` were not
  updated. 03, 04 and 05 had the same gap.
- Smells: the growing `if (name == ...)` cascade in `setParameter` and
  `unsetParameter`, and the new constructor argument threaded through 7
  places. Worth tidying before the ticket-16 port.

## Resolution (2026-10-30, with the maintainer)

- **Item 1:** the synchronous callback is the documented contract (16 README). Re-entrant render or commit from inside the callback is refused with an ERROR; this goes to ticket 14.
- **Item 2:** firing on every rank is confirmed and documented in 16. Firing on a failed render is on ticket 14 (from followups/08).
- **Item 3:** ptc always advertises `KHR_FRAME_COMPLETION_CALLBACK`; this goes to ticket 12.
- **Item 4:** covered by the 01 decision that apps must release frames before the device (16 README).
- **Item 5:** a check from inside the callback is added to ticket 13's first checkbox.
- **Item 6:** a test for unsetting only the UserData is added to ticket 14. The smells are left for the 16 port.
