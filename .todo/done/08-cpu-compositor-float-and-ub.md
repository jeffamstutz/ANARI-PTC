# 08: CPU compositor supports FLOAT32_VEC4 and is UB-free

**What to build:** The CPU compositor handles `FLOAT32_VEC4` color, as the CUDA path does. Today requesting it throws an exception out of `anariRenderFrame`. The ticket also removes undefined behavior in the CPU compositor:
- it dereferences an end iterator when sorting fragments
- it type-puns float/int via reference casts

No C++ exception may escape through any C API entry point. That includes the type-size check used when reading parameters, which can throw.

**Blocked by:** 01 (CPU variant builds again), 02 (MPI test harness in the repo)

**Status:** ready-for-agent

- [ ] CPU path composites `FLOAT32_VEC4` correctly on 1, 2 and 3 ranks
- [ ] No end-iterator dereference or aliasing-violating casts (clean under `-fsanitize=address,undefined` on 2 ranks)
- [ ] Errors surface through the status callback, not exceptions
- [ ] Harness float test for the CPU variant no longer marked expected-fail
