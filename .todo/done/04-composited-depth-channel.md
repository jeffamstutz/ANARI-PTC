# 04: Composited depth channel

**What to build:** An application that enables `channel.depth` and maps it on rank 0 receives the per-pixel minimum depth across all ranks (spec: Euclidean distance to the camera; infinity where nothing was hit). Today the CUDA path never writes depth, so mapping it returns uninitialized memory. The device also forces depth on internally and swallows the app's own `channel.depth` setting, so depth can be mapped even when the app never asked for it.

During the investigation, the smoke test's depth check printed neither PASS nor FAIL. Find out why as part of this ticket.

**Blocked by:** 02 (MPI test harness in the repo)

**Status:** ready-for-agent

- [ ] CUDA and CPU paths both produce composited depth on rank 0
- [ ] Mapping `channel.depth` is only possible when the app enabled it; otherwise it reports a status message and returns null with zero size
- [ ] The device still requests depth from the wrapped frame internally, regardless of the app's setting
- [ ] Harness checks depth on a hit pixel and on a background pixel on 1, 2 and 3 ranks
