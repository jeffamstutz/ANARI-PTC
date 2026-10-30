# 05: Frame parameter tracking is robust

**What to build:** The ptc frame always agrees with the wrapped frame about its size and color type, whatever order the app sets, unsets and commits parameters in:
- Today, unsetting `size` or `channel.color`, or calling `anariUnsetAllParameters` on the frame, leaves stale tracked state.
- The CPU path ignores a color-type change when the size is unchanged (the CUDA path was fixed for this in 0de24ec).
- Rendering without `channel.color` set writes into a null host buffer.

The tracked values should probably be taken on commit rather than on set, matching ANARI's commit semantics.

**Blocked by:** 02 (MPI test harness in the repo)

**Status:** ready-for-agent

- [ ] Size and color type take effect on `anariCommitParameters`, not on `anariSetParameter`
- [ ] Unset / unset-all reset the tracked state; the wrapped frame keeps its internal depth request
- [ ] Changing only the color type reallocates buffers on both paths
- [ ] Rendering a frame with no `channel.color` doesn't crash, and color can't be mapped
- [ ] Harness covers: resize between frames, color-type change between frames, render without color
