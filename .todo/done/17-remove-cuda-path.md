# 17: Remove the CUDA compositing path

**What to build:** ptc composites only on the CPU. Delete the CUDA frame wrapper, the vendored deepCompositing library, the CUDA build logic and the CUDA-only tests, so there is one backend and one build configuration.

**Why (decided 2026-10-30):** the CUDA path doesn't save a copy. ptc maps the wrapped frame to host memory, uploads it, composites on the GPU and copies the result back (`ptc/CUDAFrameWrapper.cu`), so only the compositing step itself gets faster. Meanwhile it is the source of most open follow-ups: GPU selection policy, context save/restore, the CUDA driver dependency and CMake 3.17 floor, 8-bit fragments capping `FLOAT32_VEC4` precision, duplicated sRGB code, and the vendored fork of Ingo's deepCompositing. This answers 15's CUDA question as "upstream CPU-only". Ingo should be told before this goes upstream (see 15).

**Blocked by:** None (can start immediately).

**Status:** done (2026-11-02)

- [x] **Timing baseline, before deleting anything.**
  - On the current tree, time compositing on the CUDA build (`~/build/claude/anari-ptc`) and the CPU build (`~/build/claude/anari-ptc_cpu`).
  - Use 3840x2160 with `FLOAT32_VEC4` and `UFIXED8_RGBA_SRGB` color plus depth, on 2 and 4 ranks. Report the median of at least 20 frames after warm-up, both for compositing alone and for the whole `anariRenderFrame`.
  - Record the numbers in this ticket's resolution.
  - The numbers are informational only: removal proceeds whatever they show (decided 2026-11-02, see 15).
- [x] Remove `ptc/CUDAFrameWrapper.*`, `ext/deepCompositing/` (including `samples/`), `tests/cudaDevice.*`, the `onegpu` tests, and the CUDA logic in both `CMakeLists.txt` files (`PTC_DISABLE_CUDA`, `USE_CUDA`, the CUDA arch selection, `find_package(CUDAToolkit)`, `CUDA::cuda_driver`, `PTC_CUDA`).
- [x] Simplify `FrameWrapper`/`CPUFrameWrapper` where an interface exists only to serve two backends. Folding `CPUFrameWrapper` into `FrameWrapper` is in scope if it makes the code simpler. Keep behavior and the test suite otherwise unchanged.
- [x] `cmake_minimum_required` matches what the build actually needs (followups/13 item 2).
- [x] README: drop the CUDA build option, the GPU selection paragraph, the CUDA context notes and `DC_VERBOSE`. State that compositing is CPU-only.
- [x] `anari-ptc` and `anari-ptc_cpu` now build the same thing. Rebuild both, plus `anari-ptc_cpu_asan` with `ASAN_OPTIONS=detect_leaks=0`, and run ctest. All pass. Don't delete any build tree.
- [x] Close in the resolution, as obsolete: followups/07 item 4 (dc destructor) and item 6 (GPU choice); followups/09 items 2, 3 and 4 (CUDA rounding, vendored edits, 8-bit fragments) and the CPU/CUDA duplication noted in item 5; followups/13 items 1–4; followups/14 items 4 and 5; followups/04 item 2 (deepCompositing fork). Remove any CUDA references that remain in 16's text.

## Resolution (2026-11-02)

Code: 07b74c8 (removal and the FrameWrapper fold), plus a follow-up commit with review fixes. The fixes restore the old destruction order (the compositor is destroyed before the wrapped device is released) and tidy some comments. All three trees were re-run after it: 139/139 each.

### Timing baseline (informational; taken on 49a0b15 before deleting anything)

- **Setup.** One machine: 48 cores, RTX 5090 + RTX 6000 Ada, Open MPI, RelWithDebInfo trees. The wrapped device was helide.
- **Benchmark.** A throwaway app rendered one semi-transparent sphere per rank at 3840x2160 with `channel.depth` on. It made 5 warm-up frames, then 25 timed frames. The camera changed slightly each frame so that helide really re-rendered.
- **"Compositing".** Temporary, uncommitted instrumentation timed this on rank 0 in `FrameWrapper::renderFrame`. It is `stage()` (the CUDA path's upload and fragment write; nothing on the CPU path) plus `composite()` (the CUDA path's composite and copy back).
- **"renderFrame".** The app's `anariRenderFrame` + `anariFrameReady(WAIT)` on rank 0, after an `MPI_Barrier`.
- All values below are medians.

| ranks | color | CUDA compositing (stage + composite) | CPU compositing | CUDA renderFrame | CPU renderFrame |
|---|---|---|---|---|---|
| 2 | FLOAT32_VEC4 | 36.6 ms (7.4 + 29.1) | 55.1 ms | 788 ms | 791 ms |
| 2 | UFIXED8_RGBA_SRGB | 21.6 ms (3.2 + 18.4) | 95.1 ms | 884 ms | 951 ms |
| 4 | FLOAT32_VEC4 | 50.0 ms (16.8 + 33.8) | 50.7 ms | 222 ms | 227 ms |
| 4 | UFIXED8_RGBA_SRGB | 29.8 ms (6.5 + 23.0) | 41.9 ms | 220 ms | 223 ms |

What the numbers show:
- helide's rendering dominates the whole frame. CPU compositing costs at most about 7% more per `anariRenderFrame` than CUDA (2 ranks, sRGB).
- CPU sRGB on 2 ranks is slower than float32 (95 vs 55 ms). That is worth a look if CPU compositing speed ever matters.
- In the CUDA runs, Open MPI printed `cuMemHostRegister ... failed` warnings (its smcuda BTL). They were harmless.

### Build and test

- **Trees.** `anari-ptc` (formerly CUDA) and `anari-ptc_cpu` now build the same thing. Neither tree was reconfigured from scratch. The `anari-ptc` library no longer links any CUDA library (`readelf -d`).
- **Results.**
  - `anari-ptc`: 139/139.
  - `anari-ptc_cpu`: 139/139.
  - `anari-ptc_cpu_asan` (`ASAN_OPTIONS=detect_leaks=0`): 139/139.
- **Leftover cache entries.** The old trees still cache `PTC_DISABLE_CUDA` and `CMAKE_CUDA_*`. These are now unused.

### Choices made

- **FrameWrapper fold.** `CPUFrameWrapper` is folded into `FrameWrapper`, which now owns the `CPUCompositor`. The virtual `allocate/stage/composite/cleanup/createCompositor/destroyCompositor` hooks are gone. The destructor resets the compositor before freeing the communicator, as the old subclass order did.
- **`cmake_minimum_required` stays 3.12.** Only `find_package(CUDAToolkit)` needed 3.17. The build now needs FindMPI's imported targets and `MPIEXEC_EXECUTABLE` (3.10) and `cmake_parse_arguments(PARSE_ARGV)` (3.7). The CUDA-only `CMP0104` policy guard was removed.
- **CI.** Beyond the ticket's list, the CI workflow lost its CUDA install step, its `cuda` matrix dimension and `CMAKE_CUDA_ARCHITECTURES`. `.github/scripts/` (the CUDA installer) is deleted. The workflow was already stale (it uses ANARI-SDK v0.11.1 and ubuntu-20.04) and was left otherwise as is.
- **Open: CI may not configure on the older Ubuntu rows.** Every CI row now builds the CPU compositor, which needs oneTBB >= 2021.6. `apt`'s `libtbb-dev` is probably older on ubuntu-22.04 (about 2021.5) and ubuntu-20.04 (about 2020.1). If so, those 4 of 5 rows fail to configure; before this change they took the CUDA path and never needed TBB. This was not checked on the runners. Decide whether to drop those rows or install a newer oneTBB.
- **Not done from the review: one shared color-type table.** The standards review suggested a single table for `colorFormat()`, `colorPixelSize()` and `isCompositableColorType()`. Not done; it is left for 16's port.
- **Float32 blend tolerance.** The test keeps its 1/255 tolerance for float32 ("test suite otherwise unchanged"). Only the comment that blamed CUDA's 8-bit fragments changed. Tightening it is now possible; see followups/09 item 4 below.

### Follow-ups closed as obsolete

- followups/07 item 4 (vendored `dc::Compositor` destructor) and item 6 (CUDA GPU choice).
- followups/09 item 2 (CUDA `make_8bit` rounding), item 3 (vendored edits) and item 4 (8-bit fragments for FLOAT32_VEC4 on CUDA). Item 4's second question stays open: whether to tighten the CPU float32 tolerance, which no longer needs to allow for CUDA.
- followups/09 item 5: the CPU/CUDA duplication of the format mapping, the sRGB helpers and the test format choice.
- followups/13 items 1 to 4:
  - item 1: GPU selection policy.
  - item 2: the driver dependency and the CMake minimum (resolved as above).
  - item 3: the GPU banner.
  - item 4: the GPU edge cases.
- followups/14 item 4 (vendored deepCompositing edits and `DC_VERBOSE`) and item 5 (GPU message severity: only the CPU DEBUG message remains).
- followups/04 item 2 (the deepCompositing fork).

### Ticket 16

The only CUDA mention left in 16 is the "Blocked by" entry naming this ticket by its title. It was kept as a dependency record.
