# Follow-ups from 01 (CPU variant builds again)

Commits reviewed: 9000717, f49e776, 44fe13f. 9000717 covers what the task
doc asked for. The two later commits go beyond it and include choices the
maintainer should confirm.

## 1. Scope expansion: frames are now actually destroyed (f49e776)

severity: non-blocking

`FrameWrapper::retain/release` now use `helium::RefType::PUBLIC`. Before
this, `anariRelease` on a ptc frame never destroyed it, in either variant.
This is a real bug fix. It also means `~FrameWrapper`, `~CPUCompositor` and
`~CUDAFrameWrapper` run for the first time, both mid-run and at teardown.
None of this was in ticket 01's scope. Confirm you want it to land here
rather than as its own ticket. The only coverage so far is the
implementer's manual smoke run (1 and 2 ranks, CPU and CUDA).

## 2. Release-order hazard is now reachable (consequence of #1)

severity: non-blocking (fold into ticket 11)

Frames don't retain the `PTCDevice`. The destroy callback captures raw
`this` (`ptc/PTCDevice.cpp:99-101`). If an app releases the device before
its frames:
- `~PTCDevice` may call `MPI_Finalize`. Later frame destruction then runs
  compositor teardown after finalize.
- `removeFrameWrapper` is called on a freed `PTCDevice` (use-after-free).

Before f49e776, frames leaked, so these paths could not run. Now they can.
Ticket 11 already owns device/MPI lifetime, and its "release the first
device, keep rendering with the second" test is next to this case. Decide
whether 11 should also make frames retain their `PTCDevice` (or otherwise
define the release order), and add that to 11's checklist.

## 3. TBB minimum raised to 2021.6 and process-wide `tbb::finalize` (f49e776, 44fe13f)

severity: non-blocking (record against ticket 15's TBB item)

To fix a segfault at exit (TBB workers outliving the dlclose'd device
library), `CPUCompositor`:
- holds a `tbb::task_scheduler_handle`,
- calls `tbb::finalize(handle, std::nothrow)` in its destructor,
- needs `find_package(TBB 2021.6 REQUIRED)`.

So the implementer, acting alone:
- picked a minimum oneTBB version (this drops older distro TBBs and legacy
  TBB 2020),
- committed the CPU compositor more firmly to TBB-specific lifetime APIs.

Ticket 15 still has "keep TBB required/optional, or replace with std
threading" open. If 15 drops TBB, this workaround goes away. If 15 keeps
it, confirm 2021.6 is an acceptable floor for ANARI-SDK (ticket 16). Also
note: there is one handle per compositor, i.e. per frame, not per device.
`finalize` is best-effort and returns without waiting if the host app or
the wrapped device also uses TBB, so the exit crash can still happen in
that setup.

## 4. Bookkeeping

severity: non-blocking

- The checkboxes and `Status:` in `01-cpu-variant-builds.md` were not
  updated.
- The smoke executable still exits 1 because of the missing
  KHR_DATA_PARALLEL_MPI extension (ticket 03). The render checks pass.
- `CPUFrameWrapper::updateSize` duplicates the base class's size/type
  early-out check. This is cosmetic.

## Resolution (2026-10-30, with the maintainer)

- **Item 1:** accepted as landed.
- **Item 2:** documented only. Apps must release frames before the device, and 16's README list now says so. No change to ticket 11.
- **Item 3:** recorded as input on ticket 15's TBB item. No code change.
- **Item 4:** superseded. 03 fixed the extension exit code, and 05 replaced `updateSize` with `allocate()`.
