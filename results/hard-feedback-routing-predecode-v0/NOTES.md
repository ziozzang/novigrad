# Pre-decode protocol v0

This directory preserves the first prepared protocol and its exact source and
test snapshots. No language-model decoding or outcome measurement occurred
under this protocol.

Before evaluation, review found that `verify` checked selected new LM
generations but did not replay all 64 cases through the cached environment,
host posterior, and hard-prefix path. The active v1 protocol adds that guard.
This is a pre-measurement reproducibility correction, not a response to hard
routing metrics.

The hashes of `hard_feedback_routing.py` and
`test_hard_feedback_routing.py` match their entries in the preserved
`protocol-lock.json`.
