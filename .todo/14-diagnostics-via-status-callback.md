# 14: Diagnostics go through the status callback

**What to build:** A normal run of an application using ptc produces no stdout/stderr output from the device. All diagnostics go through the ANARI status callback at an appropriate severity. Current offenders: the CPU compositor's startup `printf`, the frame wrapper's unsupported-channel `cout`, the `PRINT`/`PING` macros, and deepCompositing's `#brix.mpi` banner, per-frame "bytes in/out" lines and "NOT AFFINITIZED" warning.

**Blocked by:** 01 (CPU variant builds again)

**Status:** ready-for-agent

- [ ] No `printf`/`std::cout` in device or compositor code paths, except behind an explicit opt-in debug switch
- [ ] Former messages that carry useful info are reported at `DEBUG`/`INFO`/`WARNING` severity
- [ ] `PRINT`/`PING` debug macros removed
- [ ] Harness run on 2 ranks emits no device output on stdout
