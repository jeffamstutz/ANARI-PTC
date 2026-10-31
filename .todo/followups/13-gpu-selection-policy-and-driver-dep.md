# Follow-ups from 13 (CUDA compositor hygiene)

Commits reviewed: 29a3d0d, 256ea7d. All six checkboxes are met:
- **App's current CUDA device unchanged.** The cause was
  `createCompositor()` calling deepCompositing's `affinitizeGPU()`, which
  called `cudaSetDevice()` and never restored it. ptc now picks the GPU
  itself and sets `affinitizedGPU` without calling `affinitizeGPU()`. A
  `CurrentGPU` guard makes ptc's GPU current around allocate, stage,
  composite, cleanup and compositor destruction. It then restores the
  previously current *context* through the driver API, including "no
  context". `ptcSmoke`, in CUDA builds, makes GPU `(rank+1) % count`
  current before loading ptc. It checks that this GPU is still current
  after frame creation, after every commit and render, inside the
  completion callback (followups/06 item 5), and after release.
- **One `prepare()` per frame.** Already true since afb8f83. The only call
  is in `stage()`.
- **Non-root ranks skip the color and depth output copies.** Both copies
  are now under `if (m_rank == 0)`.
- **Harness on 2 ranks with 2 GPUs, and with every rank on one GPU.** This
  machine has two GPUs (RTX 5090 and RTX 6000 Ada). The new CUDA-only
  `ptc_smoke_onegpu_*` tests set `CUDA_VISIBLE_DEVICES=0`. The CUDA tree
  passes 137/137 and the CPU tree 130/130.
- **The `(const float3&)fragColor` cast** was already gone (56c8c5d).
- **followups/11 item 5 (CUDA GPU selection runs collectives before the
  agreement).** Its only collective is now `MPI_Comm_split_type(SHARED)`.
  It runs before anything that can throw. The "no CUDA device" throw comes
  after it, so `setCommunicator()`'s agreement covers it. That item can be
  closed.

Nothing here blocks 14. Item 1 is the decision that followups/07 item 6
asked you to make *before* 13. It was not added to the ticket, so the
implementer made it alone. Decide it before 16 writes the README.

## 1. GPU selection policy was chosen without you (followups/07 item 6)

severity: non-blocking (decide before 16)

followups/07 item 6 said: "**Decide** for 13 whether the GPU is chosen by
node-local rank in `MPI_COMM_WORLD`, by node-local rank in the frame's
communicator, or not by ptc at all (leaving it to the wrapped device). Add
that choice to 13 so its implementer doesn't make it alone." Ticket 13
never got that line. The implementer kept the existing policy:
- The GPU is this process's rank among the *first* communicator's ranks on
  its node, modulo the number of visible GPUs.
- Later communicators keep that GPU. Split communicators can therefore
  oversubscribe a GPU (followups/07 item 6 still applies).
- The node-local rank now comes from `MPI_Comm_split_type(SHARED)` instead
  of deepCompositing's hostname `Allgather`. These are equivalent on normal
  clusters, but this is still a mechanism change.

The README now documents this policy as fact ("a frame's first
communicator picks the GPU..."), so 16's device README would inherit it.

The ticket's new test exposes a fourth option: the app has made a GPU
current before using ptc. ptc could composite on *that* device, which is
the device the wrapped renderer most likely uses. That would avoid
device-to-device copies, and ptc would not need to pick a GPU at all.

**Decide** between:
- node-local rank in the first communicator (current behavior);
- node-local rank in `MPI_COMM_WORLD`;
- the app's current device at first commit;
- a parameter, for example a `cudaDevice` on the device or frame.

Then update the README to match.

## 2. New link dependency on the CUDA driver library, and a stale CMake minimum

severity: non-blocking (15/16)

- ptc now links `CUDA::cuda_driver` (`libcuda`), because it uses
  `cuCtxGetCurrent`/`cuCtxSetCurrent` to restore the app's context. Before
  this, it needed only the runtime. The runtime-only alternative is to
  save and restore the *device* with `cudaGetDevice`/`cudaSetDevice`. That
  is simpler, but it can create a primary context on GPU 0 for an app that
  had none, and it loses the distinction between non-primary contexts.
- `find_package(CUDAToolkit)` (in `ptc/` and `tests/`) needs CMake 3.17.
  The top-level and vendored `cmake_minimum_required` still say 3.12. So a
  CUDA build on CMake 3.12 to 3.16 now fails at configure time, even though
  the declared minimum allows those versions.

**Confirm** the driver-API dependency, which feeds 15's "CUDA compositing
upstream" decision. **Raise** `cmake_minimum_required` to 3.17 (or to the
SDK's minimum, ahead of 16), or guard the CUDAToolkit use.

## 3. Overlap with 14: GPU banner replaced by an INFO status

severity: non-blocking (ticket 14)

deepCompositing's `#brix.mpi ... GPU #n` banner and its "NOT AFFINITIZED"
warning both live in `affinitizeGPU()`. ptc no longer calls that, so
neither prints any more. Instead, ptc reports `compositing on GPU N
(name)` once per frame at INFO. That is 14's job, and the implementer chose
the wording and severity. The vendored `affinitizeGPU()` and its `printf`s
are now dead code from ptc's point of view.

**Note in 14** that those two offenders are unreachable. Then decide
whether to delete them from the vendored copy, which ties into 15's "local
fork" question, or leave them. **Confirm** INFO for the GPU message, or
move it to DEBUG.

## 4. Edge cases left open

severity: non-blocking

- **A rejected communicator still fixes the GPU.** `m_gpu` is set in
  `createCompositor()` before `setCommunicator()`'s agreement. If the
  agreement rejects that communicator, the next one reuses the pick. So the
  GPU comes from the first *attempted* communicator, not the first accepted
  one. This matters only if item 1 keeps "first communicator".
- **Redundant save/restore.** The vendored `Compositor::finish`, `prepare`
  and `resize` still save and restore the device around their own work.
  Under `CurrentGPU` this is harmless.
- **The current-device check is vacuous on one GPU.** With one visible GPU,
  `(rank+1) % 1 == 0` is also ptc's pick, so the `onegpu_*` tests cannot
  catch a leak. Only the multi-GPU runs exercise the check. CI without
  two GPUs would not catch a regression.

## 5. Bookkeeping

severity: non-blocking

- The checkboxes and `Status:` in `13-cuda-compositor-hygiene.md` were not
  updated. Earlier tickets had the same gap.
- `tests/cudaDevice.{cpp,h}` exists only because `<cuda_runtime.h>` and
  `anari::math` clash on `uint2`/`float3`. Keep this in mind when 16 merges
  the harness helpers (followups/11 item 6).
