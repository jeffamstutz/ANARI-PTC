# Follow-ups from 02 (MPI test harness in the repo)

Commits reviewed: df422ad, 7697907. All five checkboxes in the task doc
are met by the commits. I checked the CPU tree's
`~/build/claude/anari-ptc_cpu/Testing/Temporary/LastTest.log`: ufixed8 passes
on 1/2/3 ranks, float32 passes on 1/2/3 ranks only through the "not
implemented" regex, and the negative test passes. Nothing here blocks a
queued ticket. The items below are for you to confirm or act on.

## 1. Stale installed ptc library in ~/opt/anari/lib

severity: non-blocking

`~/opt/anari/lib/libanari_library_ptc.so` is dated Nov 10 2025 and sits
next to helide. Any process that resolves `anari_library_ptc` through
`~/opt/anari/lib`, such as a manual `mpiexec` run with that directory on
`LD_LIBRARY_PATH`, gets the old device. The harness now guards against this
in two ways: the build dir goes first on the path, and the loaded library is
checked with `--expect-library`. The file is outside the repo, so it is your
call whether to delete it, reinstall from this tree, or keep it.

Consequence for 01: its manual smoke runs (the "CPU path renders on 1 and 2
ranks" checkbox) may have loaded this stale copy. The new ctest run is
against the fresh library and covers that checkbox for ufixed8 on 1, 2 and 3
ranks, so 01's claim now holds. The CUDA path's 01 manual run is likewise
covered by the CUDA tree's ctest pass, according to the implementer. The
CUDA LastTest.log on disk shows only the last single-test rerun.

## 2. Harness defaults chosen by the implementer

severity: non-blocking

Confirm or override each of these. They are all easy to change later:
- `PTC_BUILD_TESTING` defaults to OFF.
- The `KHR_DATA_PARALLEL_MPI` check is soft (it prints XFAIL) unless
  `--require-mpi-extension` is passed. 03's checklist says "Harness checks
  the extension on all ranks" but does not say to turn the flag on or to
  make it the default. Consider adding that line to 03.
- The CPU float32 expected-fail is a `PASS_REGULAR_EXPRESSION` on the
  exception text. While this property is set, the exit code is ignored. Today
  the process dies through `std::terminate` (see the log), and that still
  counts as a pass. 08 already has "no longer marked expected-fail" on its
  checklist. Once 08 routes errors through the status callback, the message
  text may change. Don't leave the regex in place as a silent pass.
- The 3-rank tests use no `--oversubscribe`, so they need 3 slots. That is
  fine on this 48-core box, but Open MPI refuses on a small CI runner.
- helide is found only through `anari_DIR/../../..`, so it is not looked for
  in an in-tree SDK build. Ticket 16 ("MPI tests registered in SDK ctest")
  will need to rewire this, along with the `LD_LIBRARY_PATH` handling.

## 3. File headers on new files

severity: non-blocking (owned by ticket 15)

`tests/*` carry only `SPDX-License-Identifier: Apache-2.0`. They have no
copyright line and not the repo's boxed "Copyright Ingo Wald" banner. That
matches neither the current repo nor a decided SDK convention. Ticket 15's
header decision should cover these files too.

## 4. Bookkeeping

severity: non-blocking

- The checkboxes and `Status:` in `02-mpi-test-harness.md` were not
  updated.
- The df422ad commit message says the CPU float32 tests are `WILL_FAIL`.
  7697907 changed that to a regex match, so the message is stale. History
  only, no action needed.
- `--depth` support is in `ptcSmoke` but no test registers it. This is
  intended, for ticket 04.

## Resolution (2026-10-30, with the maintainer)

- **Item 1:** the maintainer reinstalled `~/opt/anari/lib/libanari_library_ptc.so` from `~/build/claude/anari-ptc`, so it is no longer stale. The harness's `--expect-library` guard stays.
- **Item 2:**
  - `PTC_BUILD_TESTING` stays OFF by default.
  - Open MPI `--oversubscribe` is added to ticket 14.
  - The helide lookup and `LD_LIBRARY_PATH` rewiring are noted on ticket 16.
  - The XFAIL point was superseded by 03. The float32 regex point was superseded by 08.
- **Item 3:** headers on new files are noted on ticket 15.
- **Item 4:** history only. `--depth` is now used (04).
