# 01: CPU variant builds again

**What to build:** A CPU-only configuration (`PTC_DISABLE_CUDA=ON`) configures, compiles and links. The CPU frame wrapper was not updated when the shared frame wrapper switched its host color/depth buffers from vectors to raw pointers. The build also has stale CMake entries: a `TBB:tbb` link typo, an include directory pointing at a nonexistent `submodules/deepCompositing`, and helium linked twice.

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] `cmake -DPTC_DISABLE_CUDA=ON` configures and builds cleanly with no new warnings
- [ ] CUDA configuration still builds
- [ ] TBB is linked through its imported target (`TBB::tbb`)
- [ ] No references to nonexistent include directories remain
- [ ] The CPU path renders the existing smoke scene correctly on 1 and 2 ranks (manual run is fine until 02 lands)
