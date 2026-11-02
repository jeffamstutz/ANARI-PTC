# 15: Upstream scope decisions

**What to build:** Decisions (made by the maintainer, not an agent) that gate the port into ANARI-SDK. Record each outcome in this file.

**Blocked by:** None (can start immediately).

**Status:** done (decided 2026-11-02)

- [x] File headers / copyright: current files carry "Copyright 2024 Ingo Wald" Apache-2.0 headers; the SDK uses Khronos SPDX headers. Agree the approach with Ingo. New files since the upstreaming work began (`tests/*`, `ptc/MpiExtension.*`, `ptc/WrappedLibrary.*`) carry only an SPDX Apache-2.0 line; cover them too (followups/02 item 3)
  - **Decided 2026-11-02:** Khronos SDK header only (`Copyright <years> The Khronos Group` + `SPDX-License-Identifier: Apache-2.0`) on every file, no Ingo Wald line. Tell Ingo when telling him about the port.
- [x] **Leaning CPU-only (2026-10-30): ticket 17 removes the CUDA path, subject to its timing gate. Tell Ingo before upstreaming.** CUDA compositing upstream: ship the vendored deepCompositing library (and if so, without its samples and their owl/stb copies), or upstream CPU-only first. Note that the vendored copy is now a local fork: 04 added a depth output and MPI tag to `dc::Compositor::finish`, and 08/09 change the blend. Decide whether those changes go back to Ingo's upstream deepCompositing, or whether this repo becomes the source of truth (followups/04 item 2)
  - **Decided 2026-11-02:** upstream CPU-only, unconditionally. 17 removes the CUDA path whatever its timings show; the gate only records numbers. The deepCompositing fork question is moot (the vendored copy is deleted). Still tell Ingo before upstreaming.
- [x] TBB: keep as a required/optional dependency of the CPU compositor, or replace with standard-library threading
  - Input (followups/01 item 3): the CPU compositor now requires oneTBB >= 2021.6. It holds a `tbb::task_scheduler_handle` per compositor and calls `tbb::finalize` to avoid a segfault at exit, caused by TBB workers outliving the dlclosed library. This is best-effort: it doesn't help if the host or the wrapped device also uses TBB. Replacing TBB removes the workaround; keeping it means accepting the 2021.6 floor for ANARI-SDK.
  - **Decided 2026-11-02:** keep TBB as a required dependency; accept the oneTBB >= 2021.6 floor and the best-effort `tbb::finalize` workaround.
- [x] Device/library name: keep `ptc`, or rename
  - **Decided 2026-11-02:** keep `ptc` (`anari_library_ptc`).

**2026-11-02:** Ingo approved the migration into ANARI-SDK and the license/header change.
