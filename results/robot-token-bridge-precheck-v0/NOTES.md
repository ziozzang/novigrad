# Precheck v0 archive

This directory preserves the complete first run plus the exact source and test used to produce it. A pre-publication audit found a TTL boundary bug: `CommandReceiver.rate` treated `expires_s` as inclusive (`now > expires_s`), and `accept` allowed a command whose expiry equaled the current time. At 100 Hz, the nominal 0.25 s one-shot motor pulse therefore remained active for 26 ticks (0.26 s) instead of the intended half-open 25 ticks.

The active rerun changes command validity to `[issued_s, expires_s)`. This archive is retained as a superseded measurement and must not be presented as the corrected result. Its source hashes can be checked against `protocol-lock.json`.
