# Follow-ups from 11 (MPI lifetime with multiple devices)

Commits reviewed: 5f0e01f, 166d541. All seven checkboxes are met:
- **Every device is counted.** Each device that initializes while ptc's
  private MPI is alive holds it (`m_holdsPrivateMpi`). A static mutex guards
  the count.
- **The last holder finalizes.** `~PTCDevice` decrements the count and calls
  `MPI_Finalize` at zero.
- **App-initialized MPI is never finalized.** Devices don't hold MPI that the
  app initialized.
- **Lifetime tests.** `ptcLifetime` runs `two-devices`, `single` and
  `query-release`, each with app-initialized and ptc-initialized MPI, on
  1 and 2 ranks.
- **Failure agreement.** A one-int `MPI_Allreduce(MIN)` runs before
  `composite()`/`compositeIds()`. To make this possible, compositing is split
  into a local `stage()`/`stageIds()` and the collective half.
  `ptc_smoke_fail_last_rank{,_ids}` runs on 2 and 3 ranks. These tests hung
  before the fix.

Both trees (CUDA and CPU) pass 123/123.

Nothing here blocks a queued ticket. Items 1 to 3 are choices the implementer
made alone, and they change documented or previously landed behavior. Items 4
and 5 add work to tickets 14 and 13.

## 1. Creating a device after ptc finalized MPI is now an error

severity: non-blocking (16 README)

Suppose the app doesn't initialize MPI and creates ptc devices one after
another (create, release, create). The second `initDevice()` now throws:
"MPI was finalized when the last ptc device that used it was released...".
Before, it would have called `MPI_Init` a second time, which MPI forbids.
The README now tells such apps to initialize MPI themselves.

The ticket asked for "finalize when the last device is destroyed", so this
follows from it. But it makes sequential-device use without app MPI a
documented unsupported pattern. The alternative is for ptc to never finalize
its private MPI on device release, and to finalize it at process exit
(atexit) or not at all. That would support sequential devices, but it breaks
the ticket's "finalize at last device" wording.

Two related changes:
- `acquireMpi()` now runs *first* in `initDevice()`, before the wrapped
  library loads. A device whose wrapped library fails to load (ticket 12's
  FATAL path) still initializes MPI and holds it until the device is
  released.
- Ticket 12's harness tests (selection via parameter, failure case) must
  initialize MPI in the app if they create more than one device per process.

**Confirm** "error after private finalize" as the contract for 16's README,
or choose "finalize at exit" instead.

## 2. Missing or wrong-typed wrapped depth/color now fails the whole frame (changes ticket 10)

severity: non-blocking

`renderFrame()` now maps the wrapped `channel.depth` (always) and
`channel.color` through `mapWrapped()`. That call throws unless the channel
has the expected type and the frame's size. The throw fails the agreement,
so every rank reports an ERROR and nothing is composited.

Before this change:
- `compositeIds()` warned and composited that rank's IDs as no hits. 10
  landed this, and followups/10 item 7 listed it as a known noisy path. It
  has no recorded resolution yet.
- The color compositors used the mapped pointer unchecked. On CPU this
  could crash.

Failing the frame is safer than crashing. But it means that a wrapped device
that doesn't render `channel.depth` as FLOAT32 cannot be used with ptc at
all, even for color-only compositing. Ticket 12 will let apps pick any
wrapped device. **Confirm** this, and have 16's README say that the wrapped
device must support `KHR_FRAME_CHANNEL_DEPTH`. Alternatively, decide on a
fallback, such as treating a missing depth as "no hits" or as constant
depth.

## 3. Scope extensions: `compositeIds()` and commit-time compositor creation are agreed too

severity: non-blocking

The implementer acted on two reviewer notes that the maintainer has not
resolved:
- **followups/10 item 4.** `compositeIds()` is now covered by the
  agreement. This is the obvious reading. Mark that item resolved.
- **followups/07 item 5.** `setCommunicator()` now agrees after
  `createCompositor()`. If any rank failed, every rank destroys its
  compositor, frees the dup, and throws. The frame is then left with
  `m_comm = MPI_COMM_NULL`, because the old communicator was already torn
  down. Renders report ERROR until the app sets and commits
  `mpiCommunicator` again. This is the same state that 07's "pointer to
  `MPI_COMM_NULL`" path produces (followups/07 item 3, also undecided).
  These are not covered:
  - communicator teardown, which runs collectively before the new dup;
  - `MPI_Comm_dup` itself;
  - CUDA `affinitizeGPU()` (item 5).

  None of these is tested.

**Confirm** the "failed commit leaves the frame unrenderable until another
communicator commit" contract. Close followups/07 item 5, either as "covered
for creation, teardown accepted as a limitation" or with a new checkbox.

## 4. Ticket 14 will remove the only failure the agreement tests use

severity: non-blocking (ticket 14)

`--fail-last-rank` makes the last rank commit `channel.color` as
FLOAT32_VEC3, so `updateBuffers()` throws at render on that rank only. That
is the only failure path that is tested. Ticket 14 moves this rejection to
`anariCommitParameters` with `INVALID_ARGUMENT`. After that:
- `ptc_smoke_fail_last_rank*` stops exercising the render-time agreement.
  It will either fail, or quietly test something else.
- The rejected rank's color type at render becomes whatever 14 decides,
  such as unset or the previous value. Other ranks keep a different color
  type. The agreement only checks success. It does not check that every rank
  composites the same format, so mismatched formats could then reach the
  compositor's collectives.

**Add a checkbox to 14:** keep a render-time failure injection for the
agreement tests, for example a test-only env var or parameter that makes
`stage()` throw on chosen ranks. Also decide whether a commit-rejected
color type must leave every rank in the same compositing state.

## 5. Gaps the agreement does not cover (documented in the README)

severity: non-blocking (ticket 13 for the CUDA one)

- **CUDA GPU selection.** `CUDAFrameWrapper::createCompositor()` calls
  `affinitizeGPU()`, which runs MPI collectives before the commit-time
  agreement. A failure there on one rank still hangs the others. Ticket 13
  is reworking affinitization (see also followups/07 item 6). **Add** to 13:
  whatever replaces it must not run collectives that can fail on some ranks
  only, or it must be agreed.
- **Failures inside the collective half.** These still hang. Examples: a
  CUDA error in `dc::Compositor::finish()`, an allocation inside
  `CPUCompositor::run`, the ID `MPI_Reduce` buffers. This is accepted as a
  limit in the README. **Confirm** that this limit is acceptable for 16.
- **Removed barriers.** `CPUCompositor::resize()` lost its two
  `MPI_Barrier`s. They guarded only local work, and the edit was needed to
  make `allocate()` local. This was a small behavior edit to compositor
  code. No action is needed.

## 6. Smaller items

severity: non-blocking

- **Bookkeeping.** The checkboxes and `Status:` in
  `11-mpi-lifetime-multiple-devices.md` were not updated. Earlier tickets
  had the same gap.
- **Test helper duplication.** `tests/ptcLifetime.cpp` duplicates helpers
  from `ptcSmoke.cpp`. Worth sharing them before the 16 port.
- **Code review leftovers, not acted on.** `FrameWrapper::agree` is a static
  member. Mapped channels could be RAII instead of the manual unmap lambda.
- **Misleading map warning.** After a failed render, mapping color still
  gives the misleading "rendered without channel.color" warning. This was
  already assigned to 14 (followups/08 item 2).
- **Release order.** Frames released after their device still run teardown
  after `MPI_Finalize`. This stays documented-only, per the followups/01
  item 2 decision. The new README section repeats "release a device's
  frames before the device".
