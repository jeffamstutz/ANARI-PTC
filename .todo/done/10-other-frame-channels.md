# 10: Other frame channels

**What to build:** Mapping a frame channel the ptc device doesn't composite (`channel.normal`, `channel.albedo`, `channel.primitiveId`, `channel.objectId`, `channel.instanceId`) produces a status-callback warning and a well-formed empty result. Today it prints to stdout. Optionally, composite the ID channels and normal/albedo by taking the nearest-depth fragment's value, since those don't blend.

**Blocked by:** 04 (Composited depth channel)

**Status:** done

- [x] Unsupported channels report via status callback and return null with zero size and `ANARI_UNKNOWN` type
- [x] Decide (and document in the ticket on completion) whether nearest-depth ID/normal/albedo compositing is implemented now or deferred
- [x] If implemented: harness verifies `primitiveId`/`objectId` from two ranks select the nearest fragment

**Resolution (338e66a, 62e2241):** `primitiveId`, `objectId` and `instanceId` are composited now. The nearest-depth fragment wins, and at equal depth the lower rank wins. This is done with one host-side `MPI_Reduce` to rank 0 of the frame's communicator. `normal` and `albedo` are deferred because helide can't render them, so there would be no harness coverage; mapping them is refused with a WARNING. Open items are in followups/10.
