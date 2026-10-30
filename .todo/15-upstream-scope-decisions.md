# 15: Upstream scope decisions

**What to build:** Decisions (made by the maintainer, not an agent) that gate the port into ANARI-SDK. Record each outcome in this file.

**Blocked by:** None (can start immediately).

**Status:** needs-decision

- [ ] File headers / copyright: current files carry "Copyright 2024 Ingo Wald" Apache-2.0 headers; the SDK uses Khronos SPDX headers. Agree the approach with Ingo
- [ ] CUDA compositing upstream: ship the vendored deepCompositing library (and if so, without its samples and their owl/stb copies), or upstream CPU-only first
- [ ] TBB: keep as a required/optional dependency of the CPU compositor, or replace with standard-library threading
  - Input (followups/01 item 3): the CPU compositor now requires oneTBB >= 2021.6. It holds a `tbb::task_scheduler_handle` per compositor and calls `tbb::finalize` to avoid a segfault at exit, caused by TBB workers outliving the dlclosed library. This is best-effort: it doesn't help if the host or the wrapped device also uses TBB. Replacing TBB removes the workaround; keeping it means accepting the 2021.6 floor for ANARI-SDK.
- [ ] Device/library name: keep `ptc`, or rename
