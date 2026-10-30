# 14: Diagnostics go through the status callback

**What to build:** A normal run of an application using ptc produces no stdout/stderr output from the device. All diagnostics go through the ANARI status callback at an appropriate severity. Current offenders: the CPU compositor's startup `printf`, the frame wrapper's unsupported-channel `cout`, the `PRINT`/`PING` macros, and deepCompositing's `#brix.mpi` banner, per-frame "bytes in/out" lines and "NOT AFFINITIZED" warning.

**Blocked by:** 01 (CPU variant builds again)

**Status:** ready-for-agent

- [ ] No `printf`/`std::cout` in device or compositor code paths, except behind an explicit opt-in debug switch
- [ ] Former messages that carry useful info are reported at `DEBUG`/`INFO`/`WARNING` severity
- [ ] `PRINT`/`PING` debug macros removed
- [ ] Harness run on 2 ranks emits no device output on stdout
- [ ] Unsupported `channel.color` types are rejected at `anariCommitParameters` with `ANARI_STATUS_INVALID_ARGUMENT`, not at render. The misleading "rendered without channel.color" map warning is gone in that case (decided 2026-10-30, followups/08 item 2)
- [ ] `frameCompletionCallback` fires even when the render failed, after the error is reported (decided 2026-10-30, followups/08 item 3)
- [ ] When the detected `mpiexec` is Open MPI, the multi-rank ctest targets pass `--oversubscribe`, so 3-rank tests run on runners with fewer than 3 slots (decided 2026-10-30, followups/02 item 2)
- [ ] Setting `channel.depth` to a type other than `ANARI_FLOAT32` warns at commit, instead of silently turning depth off (decided 2026-10-30, followups/04 item 5)
