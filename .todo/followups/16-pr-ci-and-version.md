# Follow-ups from 16 (port into ANARI-SDK)

Commits reviewed: ANARI-SDK `feature/ptc-device` 3493ab3 and 0020661, on
top of 467eaab. That is the current `origin/next_release` and
`upstream/next_release`. Also reviewed: anari-ptc 0f48b8b (resolution only).
Every checkbox is met:
- **The port matches b7962ee.** The reviewer ran b7962ee's sources through
  the SDK's clang-format and diffed them against the port. The only
  differences are the license headers, two "(ticket N)" comment references
  in the tests, and blank comment lines. The code is otherwise identical.
  Nothing in `src/devices/ptc` mentions CUDA, tickets or followups.
- **Headers.** Every file carries `Copyright <years> The Khronos Group` and
  the Apache-2.0 SPDX line. Ingo-origin files say 2024-2026 and newer
  files say 2026. No Ingo Wald line remains (his name appears only in the
  README's paper citation).
- **clang-format.** Every .cpp and .h is clean under the SDK's
  `.clang-format`.
- **Build option.** `BUILD_PTC_DEVICE` defaults OFF. The device needs MPI
  and TBB 2021.6, both REQUIRED, and builds as C++ only. This matches 15.
- **Tests.** They are registered under `BUILD_TESTING` and find helide as
  the in-tree target. They are skipped with a warning when helide or
  mpiexec is missing. The 139/139 pass comes from the implementer's report;
  the reviewer did not re-run it.
- **Device README.** It covers every bullet the task listed: usage and
  wrapped-device selection, `mpiCommunicator`, depth-sorted premultiplied
  "over", the transparent background, the same wrapped device type on all
  ranks, collective-but-local `anariGetProperty`, the callback semantics,
  and releasing frames before the device. It also documents the int-count
  limit.
- **Root docs.** README.md and AGENTS.md list ptc.

Nothing in the code needs fixing. The items below are choices the
implementer made alone that the PR would ship as they are.

## 1. Should the PR add a CI job for ptc?

severity: blocking

The SDK's `anari_sdk_ci.yml` does not set `BUILD_PTC_DEVICE`. So the PR
adds a device that no CI builds or tests, and the 139 tests only ever run
on this machine. The SDK already has a per-device precedent:
`.github/workflows/halcyon_ci.yml` builds halcyon with helide on
ubuntu-22.04. The implementer left CI untouched. **Decide** before opening
the PR:
- Add a `ptc_ci.yml` modelled on halcyon's. It needs MPI and oneTBB 2021.6
  or newer. ubuntu-22.04's apt `libtbb-dev` is 2021.5, so use 24.04 or
  install oneTBB separately.
- Add ptc to an existing Linux row of `anari_sdk_ci.yml`.
- Ship without CI, and say so in the PR description.

This also settles what happens to ANARI-PTC's own broken workflow
(followups/17 item 1). Its "delete it, ANARI-SDK's CI covers it" option
assumes the SDK's CI does cover it.

## 2. ptc reports its own version (1.1.0), unlike the other in-tree devices

severity: non-blocking

helide and halcyon report the SDK's version. ptc keeps its own
`PTC_VERSION_*` (1.1.0), and the README and tests rely on that. Once ptc is
in the tree, keeping it means bumping a second version by hand. **Decide**
whether to keep 1.1.0 or switch to the SDK's version, which means changing
the CMake, the tests' version checks and the README's "Device properties"
section. Reviewers on the PR may raise this.

## 3. ANARI-PTC removed from the root README's implementation list

severity: non-blocking

The root README's "(Unofficial) list of actively developed ANARI
implementations" used to link `ingowald/ANARI-PTC`. The implementer replaced
that link with the in-tree entry. The device README still credits ANARI-PTC
as its origin. This is a statement about Ingo's repository. **Decide**,
perhaps with Ingo, whether ANARI-PTC is retired (archived, with its README
pointing at the SDK) or stays listed.

## 4. Tests on macOS and Windows

severity: non-blocking

The tests set only `LD_LIBRARY_PATH`, and nothing stops them from being
registered off Linux. Configuring with `BUILD_PTC_DEVICE=ON` and
`BUILD_TESTING=ON` on macOS (where the variable is `DYLD_LIBRARY_PATH`) or
on Windows (`PATH`) would register tests that may load the wrong library or
fail. The default is OFF and AGENTS.md calls halcyon "Linux only so far",
so this may be acceptable. **Decide** between guarding the tests (or the
option) with `if (UNIX AND NOT APPLE)`, documenting the device as Linux
only, or leaving it.

## 5. Decisions carried over and still open

severity: non-blocking

The resolution lists these as left to the maintainer. The PR ships the
conservative choice for each:
- MPI `int` counts: the limit is documented, not widened (followups/08
  item 5). The task asked for this decision. The documentation is the
  default the implementer picked, not a decision anyone made.
- Float32 blend tolerance of 1/255 (followups/17 item 2).
- Shared color-type table (followups/17 item 3).
- Investigating the 2-rank sRGB timing (followups/17 item 4).

## 6. Before opening the PR

severity: non-blocking

- 15 and followups/17 item 5 say to tell Ingo about the CPU-only scope
  (CUDA removed) before upstreaming. 15 records his approval of the
  migration and the license change, but not that he was told about the
  CUDA removal.
- The branch is not pushed. `origin` is the jeffamstutz fork and
  `upstream` is KhronosGroup, and both are at 467eaab, so the branch needs
  no rebase today.
- 16's `Status:` says "done ... awaiting maintainer review", and the
  implementer ticked "Ready to open a PR against `next_release`" itself.
