# ADR 0002: Defer symmetric beta regularization

**Status:** Accepted, 2026-09-30.

## Context

The provisional beta term belongs to Critic One's objective. Under the required stop-gradient matrix, Critic One receives gradients through Y while Z is detached. A term labeled l2_y_z would therefore contribute gradients only through Y and would behave like l2_y despite suggesting symmetric regularization.

Moving Z squared into Critic Two's objective would change the research objective and is outside this correctness fix.

## Decision

Expose only none, experimental linear_y, and l2_y beta modes. Reject l2_y_z with a clear configuration error; never alias it to l2_y. A future symmetric Y/Z regularizer requires advisor approval, a separately named objective, an explicit gradient contract, and focused tests.

## Consequences

Existing l2_y_z experiment configurations are invalid before training starts. Historical smoke output from that label must not be treated as evidence for symmetric regularization. The current stop-gradient matrix remains unchanged.
