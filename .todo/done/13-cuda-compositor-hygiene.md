# 13: CUDA compositor hygiene

**What to build:** The CUDA compositing path stops interfering with the wrapped device and wasting work:
- GPU affinitization, which picks a GPU from the node-local rank, currently runs in the frame constructor. It changes the calling thread's current CUDA device underneath the wrapped device and doesn't restore it.
- The compositor's `prepare()` is called twice per frame.
- Non-root ranks copy uninitialized output into their host color buffer.

**Blocked by:** 02 (MPI test harness in the repo)

**Status:** done

- [x] The current CUDA device seen by the app/wrapped device is unchanged after frame creation, commit and render, including when checked from inside the frame completion callback (followups/06 item 5)
- [x] One `prepare()` per frame
- [x] Non-root ranks skip the output copy
- [x] Harness passes on 2 ranks with 2 GPUs and with both ranks on one GPU
- [x] The `(const float3&)fragColor` reference cast in `deepCompositing.cu` is gone (09 may already have removed it while fixing the blend; see followups/08 item 5)
- [x] Non-root ranks also skip the depth output copy that 04 added (followups/04)

Note: the duplicate `prepare()` call was already removed in 05 (afb8f83). Verify it and tick that box; there's no work left (followups/05).
