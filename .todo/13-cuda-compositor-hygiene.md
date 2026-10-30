# 13: CUDA compositor hygiene

**What to build:** The CUDA compositing path stops interfering with the wrapped device and wasting work:
- GPU affinitization, which picks a GPU from the node-local rank, currently runs in the frame constructor. It changes the calling thread's current CUDA device underneath the wrapped device and doesn't restore it.
- The compositor's `prepare()` is called twice per frame.
- Non-root ranks copy uninitialized output into their host color buffer.

**Blocked by:** 02 (MPI test harness in the repo)

**Status:** ready-for-agent

- [ ] The current CUDA device seen by the app/wrapped device is unchanged after frame creation, commit and render
- [ ] One `prepare()` per frame
- [ ] Non-root ranks skip the output copy
- [ ] Harness passes on 2 ranks with 2 GPUs and with both ranks on one GPU
