# 02: MPI test harness in the repo

**What to build:** The smoke test written during the initial investigation (`.todo/assets/smoke.cpp`, with a standalone `smoke-CMakeLists.txt`) becomes a ctest target in this repo. It runs under `mpiexec` on 1, 2 and 3 ranks, rendering through the ptc device with helide as the wrapped device. Each rank contributes one colored sphere at a distinct screen position; rank 0 checks the composited image. Later tickets add their checks to this harness.

**Blocked by:** 01 (CPU variant builds again)

**Status:** ready-for-agent

- [ ] `ctest` runs the MPI smoke test on 1, 2 and 3 ranks for whichever variant (CPU or CUDA) is configured
- [ ] The test covers both `UFIXED8_RGBA_SRGB` and `FLOAT32_VEC4` color (the CPU float case may be marked expected-fail until 08)
- [ ] The test finds the freshly built ptc library and helide without manual `LD_LIBRARY_PATH` setup
- [ ] The test fails (non-zero exit on all ranks) when a pixel check fails
- [ ] Testing is gated behind an option and skipped if `mpiexec` or helide is unavailable
