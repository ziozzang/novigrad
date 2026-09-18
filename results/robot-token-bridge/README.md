# Typed heading tokens and a bidirectional control contract

[한국어](README.ko.md) · [Research report](../../reports/neural-link/robotics.html) · [Primary-source design](../../examples/bio_bridge/ROBOTICS_BRIDGE_RESEARCH.md)

This is a deterministic **host-codec and one-dimensional kinematic test**, not a language-model, embedding, neural-circuit, or biological experiment. A host reads an explicit requested heading and observed heading, optionally encodes both as discrete IDs, decodes them, computes a steering command, and receives the next observed state. It checks engineering contracts needed by a future bridge. It does not establish that a general LM understands these IDs or can control a body.

## Method and outcomes

Ten conditions share 8 RNG seeds × 16 initial states = 128 cases each; all 1,280 episodes reset. These are paired synthetic cases, not independently trained models or biological samples. Each case lasts 3 simulated seconds at 100 Hz; its explicitly supplied goal changes at 1.5 s. The plant integrates angular rate plus a constant external disturbance sampled uniformly within ±0.6 rad/s. The host controller is `2*sin(goal-heading)`. This uses a circular comparison inspired by PFL3 studies, but is not a model of their cell types, connectivity or neural dynamics.

The observation uses heading/goal sine and cosine, source and timestamp; it also records previous actual body turn and action-copy fields. **The controller only reads heading and goal:** it does not learn a forward model or consume those extra feedback fields. Under the action-copy condition, the supplied heading instead integrates commanded motion without external disturbance. This deliberately disadvantaged state estimate tests the distinction between a command and a measured outcome.

Success is final absolute angular error <0.15 rad (about 8.59°), not staying within tolerance, reaching a physical resource, or long-term stability. Mean final error is reported alongside the thresholded count. All failures are retained; the controller and horizon were not optimized for a headline success rate.

| Condition | Success | Mean final error (rad) | Rejected command attempts |
|---|---:|---:|---:|
| continuous_true_disturbance | 55/128 | 0.2904 | 0 |
| bins16_true_disturbance | 49/128 | 0.3257 | 0 |
| bins64_true_disturbance | 57/128 | 0.3040 | 0 |
| bins256_true_disturbance | 56/128 | 0.2894 | 0 |
| bins64_actioncopy_disturbance | 9/128 | 0.9211 | 0 |
| bins64_wrongframe_disturbance | 8/128 | 1.4594 | 38528 |
| bins64_oneshot_disturbance | 4/128 | 1.4721 | 0 |
| bins64_slow2hz_disturbance | 50/128 | 0.3743 | 0 |
| bins64_true_no_disturbance | 88/128 | 0.1972 | 0 |
| bins64_actuator_no_effect_disturbance | 8/128 | 1.4594 | 0 |

Increasing token resolution does not monotonically improve thresholded success in this limited assay. The 64-bin condition reaches 57/128 versus continuous 55/128, but has *worse* mean final error (0.304 versus 0.290 rad); neither is a general superiority result. No model fit, confidence interval over animals, or language-generalization claim is attached to this comparison.

## Controls that must not be conflated

- `wrongframe`: every command is rejected because the receiver requires a body-frame command. Its 38,528 rejection records include the final observation tick. This is a schema guard, not a numerical coordinate-transform ablation.
- `actuator_no_effect`: valid commands are accepted, but plant motor gain is zero. Its trajectory matches the wrong-frame stopped-actuator trajectory while its rejection count is zero. Thus accepted commands and actual effects are separately observable.
- `true_no_disturbance`: only external angular drift is removed; motor actuation still works. The easier condition is not an action-causality ablation.
- `oneshot`: one initial motor command expires at 0.25 s, then the receiver outputs zero. This is an expired pulse, **not** a persistent goal-memory or biological one-shot-recall comparison. The later goal switch is not acted on.
- `slow2hz`: a command is recomputed every 0.5 s and held between updates. This does not benchmark an actual LM's latency.

Commands use half-open validity `[issued, expires)`, reject expired/future/duplicate sequence IDs and incompatible frames, and require finite values. This diagnostic assumes a monotonically increasing caller clock; clock-rollback recovery and a production actuator safety envelope are not implemented. Heading IDs and goal IDs occupy distinct host namespaces; the public codec accepts 4–1,000 bins. They are not any pretrained LM's vocabulary. The controller's sine gain limits its output to ±2 rad/s, so the nominal ±2.5 clipping limit never activates; this run does not validate actuator saturation handling.

The two randomly drawn goals are checked against the *initial* heading to avoid an almost exactly opposing initial direction. There is no guard against a near-opposite heading at the actual switch, where a sine controller can linger near its unstable anti-goal zero. These cases remain in the results.

## Initial failure and correction

`../robot-token-bridge-precheck-v0/` preserves the first nine-condition run and matching source/test snapshots. Review found an inclusive expiry comparison: a nominal 0.25 s pulse acted for 26 integration intervals, or 0.26 s. The corrected source uses half-open expiry and has explicit boundary tests. Its one-shot success count remains 4/128, while mean final error changes from 1.4728897 to 1.4720602 rad. The earlier run's hashes were written after outcomes and are replay provenance, not preregistration.

For the corrected run, configuration and source/test hashes were written to `protocol-lock.json` before outcome calculation; `run-manifest.json` then pins that unchanged lock and results. **The corrected protocol is a post-review follow-up informed by the first run**, not an independent blind holdout. The misleading no-effect-disturbance name was clarified, a true zero-actuation control was added, and token-namespace bounds were checked. No learning occurred in either run.

## Replay and artifacts

Python 3.11.15 with NumPy 2.4.6 produced the original exact replay. `requirements.txt` pins the single third-party runtime dependency. The full raw results are stored losslessly as deterministic gzip to avoid a 68 MB raw JSON in Git. Restore them without deleting the compressed source, then verify:

```sh
python3 - <<'PYRESTORE'
from pathlib import Path
import gzip, hashlib, json
p = Path('results/robot-token-bridge/results.json')
raw = gzip.decompress(p.with_suffix('.json.gz').read_bytes())
lock = json.loads(p.with_name('run-manifest.json').read_text())
assert hashlib.sha256(raw).hexdigest() == lock['results_sha256']
if p.exists():
    assert p.read_bytes() == raw
else:
    p.write_bytes(raw)
PYRESTORE
.venv/bin/python examples/bio_bridge/robot_token_bridge.py verify
PYTHONPATH=examples/bio_bridge .venv/bin/python -m unittest test_robot_token_bridge -q
```

Verification checks the pre-outcome config/source lock, both artifact hashes, and all traces by independent recomputation. The archive preserves a failed preliminary run rather than replacing its history. `runtime.json` records the original environment after measurement; cross-version or cross-platform bit identity is not guaranteed. Eight new contract tests and all 167 bio-bridge tests passed.

The proposed **embedding/LM read adapter and grounded write adapter are not implemented by this diagnostic**. The separate [FlyGym CPU check](../robot-simulator-smoke/README.md) exercises a real physics simulator, but does not connect its observations or controls to this host-token loop or a language model. Joining those components with paired training and observation-matched baselines remains the next actual embodied-interface experiment.
