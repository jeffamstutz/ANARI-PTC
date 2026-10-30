# Follow-ups from 10 (Other frame channels)

Commits reviewed: 338e66a, 62e2241. The checkboxes:
- **Unsupported channels report through the status callback and return null,
  0x0 and `ANARI_UNKNOWN`.** Met. `frameBufferMap` no longer writes to
  stdout. `channel.normal`, `channel.albedo`, unknown channels, and ID
  channels that were not enabled as `ANARI_UINT32` at the last render all
  give a WARNING and an empty result. Every smoke run checks this.
- **Decide whether nearest-depth ID/normal/albedo compositing is done now or
  deferred, and document it in the ticket.** Decided, but not recorded in
  the ticket. The checkboxes and `Status:` of
  `10-other-frame-channels.md` were not updated (item 7). The decision:
  - `primitiveId`, `objectId` and `instanceId` are composited now.
  - `normal` and `albedo` are deferred, because helide can't render them.
- **Harness verifies `primitiveId`/`objectId` from two ranks select the
  nearest fragment.** Met. `ptcSmoke --ids` runs as `ids`, `nocolor_ids`,
  `ids_resize` and `ids_unset_all` on 1 to 3 ranks, and as `ids_split` on 2
  and 4. The implementer mutation-checked it: both farthest-wins and a
  higher-rank tie-break fail.

How the ID compositing works: each rank builds one record per pixel,
`[depth bits, rank, ids...]`. Ranks then do one host-side `MPI_Reduce` to
rank 0 over the frame's communicator, with a custom commutative "nearest"
`MPI_Op`. It runs after `composite()`, so CPU and CUDA share it, and neither
compositor changed.

Nothing here blocks a queued ticket. Items 1 to 3 are contracts or costs the
implementer picked alone. Items 4 to 6 add work to tickets 11, 12 and 14.

## 1. "Equal depth: lower rank wins" is now asserted for IDs, while still undecided for color

severity: non-blocking (09 item 1 and 16 README)

Followups/09 item 1 asked the maintainer whether "at equal depth, the lower
rank is in front" is a contract. That question has no recorded resolution.
Ticket 10 adopted the rule for IDs anyway:
- `nearestRecord` breaks ties toward the lower rank in the frame's
  communicator.
- 62e2241 added a "tie" sphere that every rank renders at the same place. The
  test asserts that rank 0's IDs win there.

So the rule now exists in three places (the CPU sort, the CUDA loop order,
and the ID reduction) and two tests depend on it. If the maintainer decides
equal-depth order is unspecified, both the 09 blend test and the 10 tie check
have to be relaxed.

**Decide** this together with 09 item 1. If it is a contract, ticket 16's
README states it for color and IDs.

## 2. IDs are the wrapped device's per-rank values, passed through unchanged

severity: non-blocking (16 README)

Rank 0 gets the nearest fragment's `primitiveId`/`objectId`/`instanceId`
exactly as that rank's wrapped device wrote them. ptc does not record which
rank a pixel came from. So:
- `primitiveId` is an index local to that rank's geometry, and is ambiguous
  across ranks.
- `objectId`/`instanceId` are only useful if the app assigns IDs that are
  unique across ranks. The test does this with `100 + world rank`.

Pass-through is the obvious reading. But the app can't tell which rank won a
pixel. Some uses would need that, such as picking in a distributed scene.

**Confirm** pass-through, and have ticket 16's README tell apps to make
object/instance IDs globally unique. The alternative is to also expose a
composited "owner rank" channel. That is not standard ANARI, so it would be a
ptc extension.

## 3. Per-frame cost of ID compositing

severity: non-blocking

When any ID channel is enabled, each frame costs:
- a host-side record buffer of `(2 + k) * 4` bytes per pixel, where k is the
  number of enabled ID channels;
- a map of the wrapped depth and of each ID channel;
- one `MPI_Reduce` with a user-defined op. That is a reduction tree that
  applies the op on the host at each step, not a gather;
- an `MPI_Type` and `MPI_Op` created and freed every frame.

There are also limits:
- More than `INT_MAX` pixels throws. Every rank throws the same way, so this
  does not hang.
- Below that, a single message can still pass 2 GB.

Nothing is paid when no ID channel is enabled. Unlike depth (04 item 3),
this cost is opt-in. On CUDA, the work does not use the GPU.

**Confirm** this is acceptable, or file caching the op/type, or a
GPU/compositor-integrated path, for ticket 13 or later.

## 4. `compositeIds()` is a new collective that 11's agreement step must cover

severity: non-blocking (ticket 11)

`renderFrame()` now runs `composite()` and then `compositeIds()`, and both
are collective over the frame's communicator. Ticket 11's checkbox puts a
one-int `MPI_Allreduce` before compositing, and names `allocate()` and
`composite()`. If `composite()` throws on some ranks only, the others now
also wait in the ID `MPI_Reduce`. A failure inside `compositeIds()` (for
example, allocating the record buffer) on one rank also hangs the rest.

**Add** `compositeIds()` to 11's agreement checkbox, so its implementer
gates both steps. Otherwise 11 may protect only `composite()`.

## 5. Extension list may advertise channels ptc refuses to map

severity: non-blocking (ticket 12, or 16)

ptc copies the wrapped device's extension list, at both device and library
level. With a wrapped device that supports `KHR_FRAME_CHANNEL_NORMAL` and
`KHR_FRAME_CHANNEL_ALBEDO` (VisRTX, for example), ptc advertises them but
refuses to map those channels. The `channel.normal`/`channel.albedo`
parameters are also still forwarded, so such a device renders them every
frame for nothing.

Ticket 12 already edits these lists (it adds `KHR_FRAME_COMPLETION_CALLBACK`).

**Decide** whether 12 also removes the normal/albedo channel extensions, and
whether ptc stops forwarding those parameters. The alternative is
implementing normal/albedo (item 6).

## 6. normal/albedo deferred only because helide can't render them

severity: non-blocking

The ticket left the implement-or-defer choice to the implementer, so the
deferral itself is within scope. The reason given is testability, not
difficulty. The same nearest-record reduction would carry FLOAT32_VEC3
values. Denoising apps on VisRTX are the likely users of these channels.

**Decide** whether to file a ticket for normal/albedo, which would need a
wrapped device or test double that renders them, or leave them unsupported
and document that in 16.

## 7. Smaller items

severity: non-blocking

- **Wrong-typed ID channel parameters are silently treated as off.** Setting
  an ID channel to a type other than `ANARI_UINT32` only warns at map time,
  just like depth did before 04 item 5. Add the ID channels to 14's
  "warns at commit" checkboxes for consistency.
- **`instanceId` has no check.** helide gives surfaces outside an instance no
  instance ID, so the harness skips it. The reduction code is shared with the
  other two channels, but nothing checks that it is mapped or populated.
- **Wrapped frame missing an ID or depth map.** Then every rank warns on
  every frame and composites that rank's pixels as no hits. That could be
  noisy if a wrapped device doesn't support one of the channels.
- **Test behavior changed.** Two-frame tests (`--resize`, `--first-color`,
  `--unset-all`) now enable the ID channels on the first frame whenever the
  second frame has them, or the reverse. So the reduction and the
  enable/disable switch run there, but the first frame's ID output is not
  checked.
- **Stale IDs after a failed `composite()`.** As with depth, the "rendered
  with IDs" flag is set before compositing. If `composite()` throws, a later
  map can return the previous frame's IDs. Ticket 11's error handling may
  want to clear this.
- **Bookkeeping.** The checkboxes and `Status:` in
  `10-other-frame-channels.md` were not updated, and the ticket does not
  record the implement-or-defer decision, which the ticket explicitly asked
  for. Record the decision summarized above when landing.
