# Baseline policy

DDPG is the first implementation and verification gate. Standard DDPG with online and target actor/critic is the primary four-network comparison. DDPG without a target actor helps isolate that design change. TD3 is the next strong direct continuous-control baseline.

SAC is a useful later continuous-control reference, but its stochastic policy and entropy objective change the assumptions. A2C and A3C are later on-policy contextual baselines, not architecturally equivalent to DDPG. A3C requires a separate multiprocessing design and must not be inserted as a trivial off-policy trainer variant.

Do not vendor a baseline in this stage. Later, either implement a clean, tested version under the common interface or build a thin adapter to a maintained library for verification. Any copied or adapted code needs source URL, exact commit or version, license, and local modification record in `THIRD_PARTY_NOTICES.md` before inclusion.
