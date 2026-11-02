# Follow-ups from 17 (remove the CUDA compositing path)

Commits reviewed: 07b74c8, 5e9e81d. Every checkbox is met:
- **Timing baseline.** Taken on 49a0b15 before deleting anything. 2 and 4
  ranks, 3840x2160, `FLOAT32_VEC4` and `UFIXED8_RGBA_SRGB` plus depth,
  median of 25 frames. Both compositing alone and the whole
  `anariRenderFrame` are reported. CPU costs at most about 7% more per
  frame (951 vs 884 ms, 2 ranks, sRGB).
- **Removed:** `ptc/CUDAFrameWrapper.*`, `ext/deepCompositing/` (with
  `samples/`), `tests/cudaDevice.*`, the onegpu tests, and all CUDA CMake
  logic. Outside `.todo/`, `git grep -i cuda` finds only an old changelog
  line in the README and a comment that names GPU sRGB as a reference.
- **`CPUFrameWrapper` is folded into `FrameWrapper`.** The 5e9e81d fix
  restores the old destruction order.
- **`cmake_minimum_required` stays 3.12.** The ticket's reasoning holds:
  only `CUDAToolkit` needed 3.17.
- **README** is CPU-only and says oneTBB >= 2021.6.
- **Tests:** all three trees pass 139/139 (implementer's report; not re-run
  here).
- **Follow-ups closed:** every item the ticket listed is closed in its
  resolution. Two CUDA items the ticket did not list (followups/06 item 5,
  followups/08 item 1) were already settled by 13 and 09.
- **Ticket 16:** its only CUDA mention is the "Blocked by" entry for 17.

Nothing here blocks 16. The items below are open questions the
implementer recorded or left behind.

## 1. ptc's own CI is broken on every row, and the TBB floor adds to it

severity: non-blocking

The implementer edited `.github/workflows/Ubuntu.yml`, which the ticket did
not list. The edits only remove the CUDA install step, the `cuda` matrix
dimension and the `CMAKE_CUDA_ARCHITECTURES` flags, and they delete
`.github/scripts/`. That follows directly from the removal. The implementer
also flagged that oneTBB >= 2021.6 probably fails to configure on the 22.04
and 20.04 rows. Before this change those rows took the CUDA path, which
never needed TBB.

This review found more. The workflow clones ANARI-SDK `v0.11.1`, but
`CMakeLists.txt` requires `find_package(anari 0.15.0 REQUIRED)` (since
64a45bf). So every row, including 24.04, already fails at configure,
whatever its TBB version. GitHub has also retired its `ubuntu-20.04`
runners. CI only runs on pushes to `master`, so nobody has noticed.

**Decide** which of these to do:
- Fix the workflow: bump ANARI-SDK to >= 0.15, and drop the 20.04 rows.
  For 22.04, either drop the rows or install oneTBB >= 2021.6.
- Delete the workflow, because after 16 the code is built and tested by
  ANARI-SDK's CI.
- Leave it broken until 16 lands.

## 2. Tighten the float32 blend tolerance now that CUDA is gone?

severity: non-blocking

This is the open half of followups/09 item 4. The float32 blend tests still
allow 1/255, a tolerance that existed only for CUDA's 8-bit fragments. The
implementer kept it, citing "test suite otherwise unchanged". **Decide**
whether to tighten it, here or in 16's test port.

## 3. Shared color-type table was deferred to 16, but 16 doesn't say so

severity: non-blocking

The standards review suggested one table behind `colorFormat()`,
`colorPixelSize()` and `isCompositableColorType()`. The resolution says
this "is left for 16's port", but 16's text has no checkbox or note for it.
As things stand, it will be forgotten. **Decide** whether to add it to 16
or drop it.

## 4. CPU sRGB is slower than float32 on 2 ranks

severity: non-blocking

On 2 ranks, CPU compositing takes 95 ms with `UFIXED8_RGBA_SRGB` and 55 ms
with `FLOAT32_VEC4`. On 4 ranks it is 42 ms vs 51 ms, so the 2-rank
number is the odd one out. It costs about 7% of the frame, and helide's
render time dominates. **Decide** whether to investigate (LUT decode,
`tbb` partitioning) or let it go. No ticket depends on it.

## 5. Bookkeeping

severity: non-blocking

- The `anari-ptc` and `anari-ptc_cpu` trees still cache `PTC_DISABLE_CUDA`
  and `CMAKE_CUDA_*`. They are harmless. Reconfigure from scratch if you
  want clean caches. The implementer did not, because the ticket said not
  to delete any build tree.
- The implementer set 17's `Status:` to done and ticked its own checkboxes.
- 15 still says "tell Ingo before upstreaming". The CUDA removal is now in
  the tree, so that conversation should happen before 16 opens its PR.
