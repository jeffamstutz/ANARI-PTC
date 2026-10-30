# Follow-ups from 07 (Honor mpiCommunicator)

Commits reviewed: 9184d6e, 75ecea8. All five open checkboxes are met:
- **Convention documented.** The README has a new section, with an example
  that passes `&comm` as `ANARI_VOID_POINTER`. `FrameWrapper.h` and
  `setParameter` explain the pointer-to-`MPI_Comm` reading.
- **Default `MPI_COMM_WORLD`.** It applies when the parameter is unset, a
  null pointer, or the wrong type. The wrong type also gives a WARNING.
- **Switching between frames.** `setCommunicator()` calls `cleanup()`,
  destroys the compositor and calls `MPI_Comm_free` on the old dup, all
  collectively over the old communicator. Covered by `world_to_split`,
  `split_to_world`, `split_unset_all`, `split_to_split` (congruent, so the
  compositor is kept) and `split_to_interleaved` (4 ranks, a real teardown).
- **Non-root map.** The frame returns its local buffer, whose contents are
  undefined. Every test now maps color and depth on non-root ranks too.
- **Harness.** `<color>_split` on 4 ranks splits world into two 2-rank
  communicators. Each half composites its own image to its own rank 0.

Nothing here blocks a queued ticket. Items 1 to 4 are choices the implementer
made alone. They change which calls are collective, edit vendored code, or
interpret the spec, and ticket 16's README will document them. Items 5 to 7
add work to tickets 11, 13 and 14.

## 1. Which frame calls are now collective

severity: non-blocking (16 README documents the outcome)

The ticket only asked for "collective setup in lockstep". The implementer
moved collectives between calls:
- **`anariNewFrame` is no longer collective.** The compositor is not created
  in the frame constructor any more.
- **The first `anariCommitParameters` on a frame is collective** over the
  frame's communicator, because of `MPI_Comm_dup` and compositor creation.
  The same holds for any later commit that sets or unsets `mpiCommunicator`,
  over both the old and the new communicator. Commits that do not touch
  `mpiCommunicator` stay local.
- **Frame release is collective** over the frame's communicator, because
  `~FrameWrapper` calls `MPI_Comm_free`. The free is skipped if MPI is
  already finalized. The README says this.

A side effect is that a frame created on only the ranks of a
sub-communicator works, if `mpiCommunicator` is set before the first
commit. Before 07, `anariNewFrame` was collective over world.

**Confirm** this collectiveness contract, so that ticket 16's README states
it. The alternative is to keep `anariNewFrame` collective over world and
create a world compositor eagerly. That would make a frame that only ever
uses a sub-communicator pay for a world compositor, and it would no longer
work on a subset of ranks.

## 2. ptc composites over its own `MPI_Comm_dup`, and keeps a congruent communicator

severity: non-blocking

- ptc duplicates the app's communicator. So the app's own traffic cannot
  match ptc's messages, and the app may free its `MPI_Comm` right after the
  commit. The README promises the latter, which is stronger than the
  decided convention, where the comm only had to stay valid until the commit
  returned.
- If the new communicator is `MPI_IDENT` or `MPI_CONGRUENT` to the current
  one, the compositor and its dup are kept. Nothing is torn down.

Both choices look right. **Confirm** them, mainly because the "app may free
it after commit" promise becomes part of the documented contract in 16.

## 3. A pointer to `MPI_COMM_NULL`

severity: non-blocking

This is the implementer's own interpretation. `KHR_DATA_PARALLEL_MPI` does
not cover a rank that `MPI_Comm_split` left out. The behavior is:
- The rank still takes part in tearing down the frame's previous
  communicator, so that the other ranks don't hang.
- It then reports an ERROR from the commit.
- Rendering that frame with a non-empty size gives an ERROR until another
  communicator is committed.

There is no harness test for it.

**Decide:**
- whether `MPI_COMM_NULL` is an error at all. The alternative is a no-op
  frame: render does nothing, map gives an empty result.
- whether to add a harness test, for example a 3-rank split that leaves one
  rank with `MPI_COMM_NULL`.

## 4. A destructor added to the vendored `dc::Compositor`

severity: non-blocking (adds to ticket 15's fork question)

`ext/deepCompositing/deepCompositing.{h,cu}` gains `~Compositor()`, which
frees its 4 device buffers, and deleted copy operations. Without it, a
communicator change would leak the old compositor's device memory. The
change is small and needed. But it is one more local edit to vendored
third-party code. **Add it** to 15's list of changes to the vendored copy
(after 04's depth output and MPI tag, and 08/09's blend), when deciding
whether those go back upstream.

## 5. Partial failure while creating the compositor hangs the other ranks

severity: non-blocking (ticket 11)

If `createCompositor()` or `MPI_Comm_dup` throws on some ranks only at
commit, the other ranks wait in the collective. Ticket 11's `MPI_Allreduce`
item covers `allocate()` and `composite()` only. **Decide** whether 11
extends the agreement step to commit-time compositor creation and
communicator teardown, or whether this is accepted as a known limitation.

## 6. CUDA GPU choice now happens at first commit, relative to the first communicator

severity: non-blocking (ticket 13)

`affinitizeGPU()` moved from the constructor to the first commit. It ranks
this process among the ranks of the frame's *first* communicator on its
node. A later communicator change keeps that GPU and calls `cudaSetDevice`
on it again. Consequences:
- With split communicators, two ranks on one node in different halves can
  pick the same GPU.
- The commit (not frame creation) now changes the thread's current CUDA
  device, and so does every later communicator change.

Ticket 13 already owns affinitization and the "current device unchanged
after creation, commit and render" checkbox. **Decide** for 13 whether the
GPU is chosen by node-local rank in `MPI_COMM_WORLD`, by node-local rank in
the frame's communicator, or not by ptc at all (leaving it to the wrapped
device). Add that choice to 13 so its implementer doesn't make it alone.

## 7. Wrong-typed `mpiCommunicator` warns, ahead of ticket 14

severity: non-blocking (ticket 14)

A non-`ANARI_VOID_POINTER` `mpiCommunicator` now warns and falls back to
`MPI_COMM_WORLD`. The WARNING is there because falling back on only some
ranks would hang. This is the behavior that 14 decided for `size`,
`channel.color` and `channel.depth`, but it now happens at `setParameter`,
not at commit, and it is untested. **Add** `mpiCommunicator` to 14's
wrong-type checkboxes so that 14 tests it and moves it to commit time for
consistency, or accept it as it is.

## 8. Bookkeeping

severity: non-blocking

- The checkboxes and `Status:` in `07-honor-mpi-communicator.md` were not
  updated. 03 to 06 had the same gap.
- Ticket 16's callback bullet already says "every rank of the frame's
  communicator once 07 lands". That now holds and needs no change.
- The `MPI_Comm_free` in `~FrameWrapper` is skipped after `MPI_Finalize`.
  That only matters for the unsupported release-device-first order (01
  decision).
