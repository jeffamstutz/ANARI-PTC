# 15: Upstream scope decisions

**What to build:** Decisions (made by the maintainer, not an agent) that gate the port into ANARI-SDK. Record each outcome in this file.

**Blocked by:** None (can start immediately).

**Status:** needs-decision

- [ ] File headers / copyright: current files carry "Copyright 2024 Ingo Wald" Apache-2.0 headers; the SDK uses Khronos SPDX headers. Agree the approach with Ingo
- [ ] CUDA compositing upstream: ship the vendored deepCompositing library (and if so, without its samples and their owl/stb copies), or upstream CPU-only first
- [ ] TBB: keep as a required/optional dependency of the CPU compositor, or replace with standard-library threading
- [ ] Device/library name: keep `ptc`, or rename
