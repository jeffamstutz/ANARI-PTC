# 16: Port into ANARI-SDK

**What to build:** The ptc device lives in ANARI-SDK under its devices directory and builds behind a `BUILD_PTC_DEVICE` option (default OFF) that finds MPI and oneTBB >= 2021.6 (required), per the decisions in 15. It keeps the name `ptc`, and every file carries the Khronos SDK header only. The code follows SDK conventions: clang-format, the agreed headers, C++17, the SDK's namespace and CMake helper idioms. The MPI tests run under the SDK's ctest. A device README explains:
- usage and wrapped-device selection
- the `mpiCommunicator` convention
- compositing semantics: fragments sorted by depth, then premultiplied "over" with composited alpha, matching helide and VisRTX (see followups/08)
- that each rank should render with a transparent background
- that all ranks must wrap the same device type. `anariGetProperty` is collective but answers from the local rank without communication (decided 2026-10-30, followups/03 item 3)
- frame completion callback semantics: it runs synchronously on the app thread inside `anariRenderFrame`, after compositing, on every rank (every rank of the frame's communicator once 07 lands). It fires even when the render failed, and re-entrant render or commit from inside it is refused (decided 2026-10-30, followups/06 and 08)
- that apps must release frames before releasing the ptc device. Releasing the device first is unsupported: it leads to a use-after-free and teardown after `MPI_Finalize` (decided 2026-10-30, followups/01 item 2)

**Blocked by:** 15 (Upstream scope decisions), 03 (Device reports KHR_DATA_PARALLEL_MPI), 04 (Composited depth channel), 05 (Frame parameter tracking), 06 (Frame completion callback), 07 (Honor mpiCommunicator), 17 (Remove the CUDA compositing path)

**Status:** done (implemented 2026-11-02, awaiting maintainer review)

- [x] Device builds in the SDK tree with `BUILD_PTC_DEVICE=ON`, and the SDK is unaffected with it OFF
- [x] Dependencies match decisions in 15
- [x] Sources pass the SDK's clang-format
- [x] MPI tests registered in SDK ctest and passing locally. The harness currently finds helide only via `anari_DIR/../../..` and sets `LD_LIBRARY_PATH` itself; rewire both for the in-tree SDK build (followups/02 item 2)
- [x] Device README added; root AGENTS.md device list mentions ptc
- [x] Ready to open a PR against `next_release`

Note: MPI counts in the CPU compositor are `int`. With float32 color they overflow at about 536M pixels per exchange. Decide whether to widen them or document the limit (followups/08 item 5).

## Resolution (2026-11-02)

ANARI-SDK branch `feature/ptc-device` (from `origin/next_release` 467eaab),
commits 3493ab3 and 0020661; not pushed, no PR opened.

- `src/devices/ptc/`: the device (the CPU compositor folded into the module
  target), `tests/` and `README.md`. `BUILD_PTC_DEVICE` (default OFF) in
  `src/devices/CMakeLists.txt` finds MPI and oneTBB >= 2021.6 (required).
  Khronos header only on every file; sources pass the SDK's clang-format.
  Library name `anari_library_ptc`, namespace `ptc`; it still reports its own
  version 1.1.0. Root `README.md` and `AGENTS.md` list it; the external
  ANARI-PTC link in the README's implementation list is replaced by the
  in-tree entry.
- Tests follow `BUILD_TESTING`, find helide as the in-tree
  `anari_library_helide` target (skipped with a warning without it or without
  mpiexec), and prepend `$<TARGET_FILE_DIR>` of ptc and helide to
  `LD_LIBRARY_PATH`. 139/139 `ptc_*` pass in `~/build/claude/ANARI-SDK`;
  the SDK configures and builds with `BUILD_PTC_DEVICE=OFF`.
- Code review: port is behaviour-identical to b7962ee apart from headers,
  formatting, ticket numbers in comments and build wiring. README fixes from
  review are in 0020661.

Left to the maintainer (conservative choice taken):
- MPI `int` counts: documented as a frame-size limit in the README, not
  widened.
- Float32 blend test tolerance (1/255): unchanged (followups/17 item 2).
- Shared color-type table (followups/17 item 3): not done.
- ptc's own GitHub CI (followups/17 item 1): untouched; the SDK CI does not
  build `BUILD_PTC_DEVICE`.
- Version: ptc keeps its own 1.1.0 rather than the SDK version (halcyon
  reports the SDK version).
- Tests set `LD_LIBRARY_PATH` only, so they are Linux-only as before.
