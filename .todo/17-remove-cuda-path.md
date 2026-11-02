# 17: Remove the CUDA compositing path

**What to build:** ptc composites only on the CPU. Delete the CUDA frame wrapper, the vendored deepCompositing library, the CUDA build logic and the CUDA-only tests, so there is one backend and one build configuration.

**Why (decided 2026-10-30):** the CUDA path doesn't save a copy. ptc maps the wrapped frame to host memory, uploads it, composites on the GPU and copies the result back (`ptc/CUDAFrameWrapper.cu`), so only the compositing step itself gets faster. Meanwhile it is the source of most open follow-ups: GPU selection policy, context save/restore, the CUDA driver dependency and CMake 3.17 floor, 8-bit fragments capping `FLOAT32_VEC4` precision, duplicated sRGB code, and the vendored fork of Ingo's deepCompositing. This answers 15's CUDA question as "upstream CPU-only". Ingo should be told before this goes upstream (see 15).

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [ ] **Timing baseline, before deleting anything.**
  - On the current tree, time compositing on the CUDA build (`~/build/claude/anari-ptc`) and the CPU build (`~/build/claude/anari-ptc_cpu`).
  - Use 3840x2160 with `FLOAT32_VEC4` and `UFIXED8_RGBA_SRGB` color plus depth, on 2 and 4 ranks. Report the median of at least 20 frames after warm-up, both for compositing alone and for the whole `anariRenderFrame`.
  - Record the numbers in this ticket's resolution.
  - The numbers are informational only: removal proceeds whatever they show (decided 2026-11-02, see 15).
- [ ] Remove `ptc/CUDAFrameWrapper.*`, `ext/deepCompositing/` (including `samples/`), `tests/cudaDevice.*`, the `onegpu` tests, and the CUDA logic in both `CMakeLists.txt` files (`PTC_DISABLE_CUDA`, `USE_CUDA`, the CUDA arch selection, `find_package(CUDAToolkit)`, `CUDA::cuda_driver`, `PTC_CUDA`).
- [ ] Simplify `FrameWrapper`/`CPUFrameWrapper` where an interface exists only to serve two backends. Folding `CPUFrameWrapper` into `FrameWrapper` is in scope if it makes the code simpler. Keep behavior and the test suite otherwise unchanged.
- [ ] `cmake_minimum_required` matches what the build actually needs (followups/13 item 2).
- [ ] README: drop the CUDA build option, the GPU selection paragraph, the CUDA context notes and `DC_VERBOSE`. State that compositing is CPU-only.
- [ ] `anari-ptc` and `anari-ptc_cpu` now build the same thing. Rebuild both, plus `anari-ptc_cpu_asan` with `ASAN_OPTIONS=detect_leaks=0`, and run ctest. All pass. Don't delete any build tree.
- [ ] Close in the resolution, as obsolete: followups/07 item 4 (dc destructor) and item 6 (GPU choice); followups/09 items 2, 3 and 4 (CUDA rounding, vendored edits, 8-bit fragments) and the CPU/CUDA duplication noted in item 5; followups/13 items 1–4; followups/14 items 4 and 5; followups/04 item 2 (deepCompositing fork). Remove any CUDA references that remain in 16's text.
