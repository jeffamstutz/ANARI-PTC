# 10: Other frame channels

**What to build:** Mapping a frame channel the ptc device doesn't composite (`channel.normal`, `channel.albedo`, `channel.primitiveId`, `channel.objectId`, `channel.instanceId`) produces a status-callback warning and a well-formed empty result. Today it prints to stdout. Optionally, composite the ID channels and normal/albedo by taking the nearest-depth fragment's value, since those don't blend.

**Blocked by:** 04 (Composited depth channel)

**Status:** ready-for-agent

- [ ] Unsupported channels report via status callback and return null with zero size and `ANARI_UNKNOWN` type
- [ ] Decide (and document in the ticket on completion) whether nearest-depth ID/normal/albedo compositing is implemented now or deferred
- [ ] If implemented: harness verifies `primitiveId`/`objectId` from two ranks select the nearest fragment
