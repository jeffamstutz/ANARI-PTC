# Follow-ups from 14 (diagnostics via the status callback)

Commits reviewed: 033ce84, d933f5e. All eleven checkboxes are met:
- The CPU compositor's startup `printf` is now a DEBUG status ("CPU
  compositor on rank r of n"). The `PRINT`/`PING` macros are gone.
  deepCompositing's banner, "NOT AFFINITIZED", "bytes in/out", the dump
  message and `LOG` are behind a new compile-time `DC_VERBOSE` (default 0).
  `cuda_helper.h` no longer writes to stderr. It puts the error text into
  the exception, which ptc reports as ERROR.
- `ptc_smoke_quiet_2ranks` (`tests/ExpectQuiet.cmake`) fails on any stdout
  line that `ptcSmoke` did not write.
- An unsupported `channel.color` is rejected at commit with ERROR and
  `INVALID_ARGUMENT`. The completion callback fires after a failed render.
  Re-entrant render or commit from the callback is refused. Wrong-typed
  `size`, `channel.color` and `channel.depth` warn at commit. Unsetting only
  the user data is tested. Open MPI gets `--oversubscribe`.

The CPU tree passes 139/139 and the CUDA tree 146/146. Nothing here blocks
16 (which is gated on 15 anyway). The items below are choices the
implementer made alone, and 16's device README would inherit them.

## 1. What "rejected at commit" means for an unsupported color type

severity: non-blocking (decide before 16 writes the README)

The ticket said the type is "rejected at `anariCommitParameters`". The
implementer chose this:
- The commit reports ERROR with `INVALID_ARGUMENT`, once.
- The bad type **stays committed**. It is still forwarded to the wrapped
  frame, so the wrapped device keeps rendering with it.
- Every later `anariRenderFrame` on that frame fails with another ERROR
  (`UNKNOWN_ERROR`), on every rank, until a supported type is set or
  `channel.color` is unset.

The alternatives were to drop the value, so the frame renders as if
`channel.color` were unset, or to keep the previously committed type.

The implementer kept the value partly so that `--fail-last-rank` still
fails at render on one rank and still exercises the render-time agreement.
followups/11 item 4 asked you to add a test-only failure-injection hook to
14 for that purpose. That checkbox was never added, so the test need now
shapes the API semantics. This also answers 11 item 4's question "must a
commit-rejected color type leave every rank in the same compositing state?"
implicitly. It does, because every rank refuses to render.

**Confirm** this semantics, or pick "drop and treat as unset". If you pick
that, the agreement tests need a separate injection hook (11 item 4).
**Confirm** also that a render that fails because of a commit-time
rejection should report `UNKNOWN_ERROR`. `INVALID_ARGUMENT` or
`INVALID_OPERATION` would be more specific.

## 2. Status codes and severities chosen for the new diagnostics

severity: non-blocking

followups/08 item 3 and followups/12 item 4 left the status-code policy
open. 14 needed a per-message code, so it added
`PTCDevice::reportStatus(severity, code, msg)`, a `StatusError` exception
that carries a code, and `defaultStatusCode()` for everything else. The
codes it chose:
- wrong-typed tracked parameter: WARNING with `INVALID_ARGUMENT`, once at
  the next commit. A later set or unset replaces or drops the pending
  warning.
- unsupported color at commit: ERROR with `INVALID_ARGUMENT`.
- render or commit from inside the frame's own callback: ERROR with
  `INVALID_OPERATION`.
- everything else: ERROR and above with `UNKNOWN_ERROR`, lower severities
  with `NO_ERROR` (the 12 fix).

`ANARI_UNKNOWN` as a depth or ID channel type means "off", with no
warning. **Confirm** these, especially WARNING (not ERROR) for wrong-typed
parameters, since the value is silently ignored.

## 3. Scope extensions taken from earlier follow-ups

severity: non-blocking

Earlier follow-ups asked you to either add these to 14 or accept the
status quo. They were not added to the ticket, and the implementer did
them anyway:
- **ID channels** warn at commit when wrong-typed (followups/10 item 7).
- **`mpiCommunicator`**: the wrong-type warning moved from `setParameter`
  to commit time, and it is now tested (followups/07 item 7). It still
  falls back to `MPI_COMM_WORLD`.
- **`frameCompletionCallback` and its user data** also warn when
  wrong-typed. No follow-up asked for this.
- **Every channel map is refused after a failed render**, not only color.
  Before, a failed render could still return stale depth or IDs. The map
  returns null with a WARNING that says the last render failed.

These all look consistent with the decisions already made. **Accept**
them, or say which to revert. The README's "Diagnostics" section now
documents all of them.

## 4. Edits to the vendored deepCompositing

severity: non-blocking (feeds 15's "local fork" decision)

- `ext/deepCompositing/cuda_helper.h` was largely rewritten. The macros
  throw a `std::runtime_error` with the full message instead of printing to
  stderr. They are `do { } while (0)`. `CUDA_SYNC_CHECK` now checks
  `cudaDeviceSynchronize()`'s return value. The `*_NOTHROW` variants
  (including one that called `exit(2)`) are deleted. Nothing used them.
- `deepCompositing.cu` gained a `DC_VERBOSE` preprocessor switch around
  its stdout output. `LOG` and the dump message were guarded, not deleted.
  The `affinitizeGPU()` banner and "NOT AFFINITIZED" are unreachable from
  ptc anyway (followups/13 item 3).

This adds to the local-fork divergence that 15 must decide on. Choose
between upstreaming these changes to Ingo's deepCompositing, deleting the
dead output instead of guarding it, or keeping them as they are.

`DC_VERBOSE` is only a preprocessor define. CMake has no option for it.
The README says "built with `-DDC_VERBOSE=1`". That reads like a CMake
option, but passing it to `cmake` does nothing. It has to go into
`CMAKE_CUDA_FLAGS`. **Decide** whether to add a CMake option, fix the
README wording, or drop the switch when 16 ports the code.

## 5. GPU message severity is still inconsistent (followups/13 item 3)

severity: non-blocking

The CPU compositor's startup message is DEBUG. The CUDA "compositing on GPU
N (name)" message is still INFO. 13 item 3 asked you to confirm INFO or
move it to DEBUG. 14 did not resolve that, so the two paths now disagree.
**Decide** one severity for both.

## 6. What the quiet test does not cover

severity: non-blocking

- The "What to build" text says no stdout **or stderr** output. The
  checkbox and `ExpectQuiet.cmake` only check stdout. Nothing in ptc or the
  vendored library writes to stderr now. The test does capture stderr but
  does not check it, so a regression would go unnoticed.
- The 2-rank quiet test runs only the configuration of the build tree.
  The CPU tree covers the CPU compositor and the CUDA tree covers
  deepCompositing, but only if both trees run in CI.
- Some output is still unguarded but opt-in: deepCompositing's
  frame-dump `save()` (when `dumpFileName` is set) and kernel `printf`s
  behind a hard-coded `dbg = 0`.

**Decide** whether to extend `ExpectQuiet.cmake` to stderr.

## 7. Bookkeeping

severity: non-blocking

- The checkboxes and `Status:` in `14-diagnostics-via-status-callback.md`
  were not updated. Earlier tickets had the same gap.
- The README got a new "Diagnostics" section and an `--oversubscribe`
  sentence. 16 owns the device README, so carry these over when porting.
- Review smells that were not acted on: the repeated
  `ok ? bit_cast : default` pattern, and a constructor parameter
  `reportStatus` that shadows the member function.
