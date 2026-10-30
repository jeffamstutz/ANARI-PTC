# 16: Port into ANARI-SDK

**What to build:** The ptc device lives in ANARI-SDK under its devices directory and builds behind a `BUILD_PTC_DEVICE` option (default OFF) that finds MPI and optionally CUDA, per the decisions in 15. The code follows SDK conventions: clang-format, the agreed headers, C++17, the SDK's namespace and CMake helper idioms. The MPI tests run under the SDK's ctest. A device README explains:
- usage and wrapped-device selection
- the `mpiCommunicator` convention
- compositing semantics: fragments sorted by depth, then premultiplied "over" with composited alpha, matching helide and VisRTX (see followups/08)
- that each rank should render with a transparent background

**Blocked by:** 15 (Upstream scope decisions), 03 (Device reports KHR_DATA_PARALLEL_MPI), 04 (Composited depth channel), 05 (Frame parameter tracking), 06 (Frame completion callback), 07 (Honor mpiCommunicator)

**Status:** ready-for-agent

- [ ] Device builds in the SDK tree with `BUILD_PTC_DEVICE=ON`, and the SDK is unaffected with it OFF
- [ ] Dependencies match decisions in 15
- [ ] Sources pass the SDK's clang-format
- [ ] MPI tests registered in SDK ctest and passing locally
- [ ] Device README added; root AGENTS.md device list mentions ptc
- [ ] Ready to open a PR against `next_release`

Note: MPI counts in the CPU compositor are `int`. With float32 color they overflow at about 536M pixels per exchange. Decide whether to widen them or document the limit (followups/08 item 5).
