#!/usr/bin/env python3
"""Matched-stream reliability experiment for circular heading fusion.

This is an engineered software experiment inspired by navigation problems. It
does not implement or identify a biological circuit. Ground truth is retained
by the simulator/evaluator and never enters the heading filters.
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from navigation import circular_error, wrap_degrees


ROOT = Path(__file__).resolve().parents[2]
SEEDS = tuple(range(1001, 1021))
STEPS = 120
SCENARIOS = (
    "clean", "abrupt_visual_bias", "dropout", "gradual_visual_drift",
    "odometry_bias", "simultaneous_biases",
)
ALGORITHMS = ("fixed_trust", "oracle_reliability", "no_vision", "innovation_gate")


@dataclass(frozen=True)
class Constants:
    visual_gain: float = 0.65
    oracle_bad_reliability: float = 0.0
    innovation_accept_degrees: float = 18.0
    innovation_reject_degrees: float = 35.0
    innovation_min_gain: float = 0.0
    visual_noise_sd: float = 2.5
    odometry_noise_sd: float = 0.8
    process_noise_sd: float = 0.35
    abrupt_bias_degrees: float = 95.0
    gradual_drift_degrees_per_step: float = 1.15
    odometry_bias_degrees_per_step: float = 2.2
    event_start: int = 35
    event_stop: int = 85
    warmup_steps: int = 15


@dataclass(frozen=True)
class SensorInputs:
    """Inputs visible to filters; contains no ground-truth heading."""

    visual: tuple
    odometry: tuple
    oracle_reliability: tuple


@dataclass(frozen=True)
class TrialData:
    sensors: SensorInputs
    truth: tuple


def generate_trial(scenario, seed, constants=Constants(), steps=STEPS):
    """Generate one stream once so every algorithm receives matched inputs."""
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario: {scenario}")
    if steps < 2:
        raise ValueError("steps must be at least 2")
    rng = np.random.default_rng(seed)
    truth = np.empty(steps)
    truth[0] = -110.0
    true_delta = np.zeros(steps)
    for step in range(1, steps):
        true_delta[step] = 3.2 * np.sin(step * 0.19) + rng.normal(0, constants.process_noise_sd)
        truth[step] = float(wrap_degrees(truth[step - 1] + true_delta[step]))

    visual_noise = rng.normal(0, constants.visual_noise_sd, steps)
    odometry_noise = rng.normal(0, constants.odometry_noise_sd, steps)
    visual = np.asarray(wrap_degrees(truth + visual_noise), dtype=float)
    odometry = true_delta + odometry_noise
    odometry[0] = 0.0
    reliability = np.ones(steps)
    active = np.arange(steps) >= constants.event_start
    window = active & (np.arange(steps) < constants.event_stop)

    if scenario == "abrupt_visual_bias":
        visual[window] = wrap_degrees(visual[window] + constants.abrupt_bias_degrees)
        reliability[window] = constants.oracle_bad_reliability
    elif scenario == "dropout":
        visual[window] = np.nan
        reliability[window] = 0.0
    elif scenario == "gradual_visual_drift":
        drift = np.maximum(0, np.arange(steps) - constants.event_start) * constants.gradual_drift_degrees_per_step
        visual = np.asarray(wrap_degrees(visual + drift), dtype=float)
        reliability[active] = constants.oracle_bad_reliability
    elif scenario == "odometry_bias":
        odometry[active] += constants.odometry_bias_degrees_per_step
    elif scenario == "simultaneous_biases":
        visual[window] = wrap_degrees(visual[window] + constants.abrupt_bias_degrees)
        reliability[window] = constants.oracle_bad_reliability
        odometry[active] += constants.odometry_bias_degrees_per_step

    visual_values = tuple(None if np.isnan(x) else float(x) for x in visual)
    sensors = SensorInputs(visual_values, tuple(float(x) for x in odometry), tuple(float(x) for x in reliability))
    return TrialData(sensors, tuple(float(x) for x in truth))


def _innovation_gain(residual, constants):
    magnitude = abs(residual)
    if magnitude <= constants.innovation_accept_degrees:
        return constants.visual_gain
    if magnitude >= constants.innovation_reject_degrees:
        return constants.innovation_min_gain
    fraction = (constants.innovation_reject_degrees - magnitude) / (
        constants.innovation_reject_degrees - constants.innovation_accept_degrees
    )
    return constants.visual_gain * fraction


def estimate_headings(algorithm, sensors, constants=Constants()):
    """Fuse only visual and odometry inputs; no truth argument is accepted."""
    if algorithm not in ALGORITHMS:
        raise ValueError(f"unknown algorithm: {algorithm}")
    lengths = {len(sensors.visual), len(sensors.odometry), len(sensors.oracle_reliability)}
    if len(lengths) != 1 or not lengths.pop():
        raise ValueError("sensor arrays must have equal, nonzero length")
    first = sensors.visual[0]
    if first is None or not np.isfinite(first):
        raise ValueError("a finite initial visual heading is required")
    estimate = float(wrap_degrees(first))
    output = [estimate]
    for step in range(1, len(sensors.visual)):
        delta = sensors.odometry[step]
        if not np.isfinite(delta):
            raise ValueError("odometry must be finite")
        predicted = float(wrap_degrees(estimate + delta))
        visual = sensors.visual[step]
        gain = 0.0
        if visual is not None:
            if not np.isfinite(visual):
                raise ValueError("visual headings must be finite or None")
            residual = float(circular_error(visual, predicted))
            if algorithm == "fixed_trust":
                gain = constants.visual_gain
            elif algorithm == "oracle_reliability":
                reliability = sensors.oracle_reliability[step]
                if not np.isfinite(reliability) or not 0 <= reliability <= 1:
                    raise ValueError("oracle reliability must be in [0, 1]")
                gain = constants.visual_gain * reliability
            elif algorithm == "innovation_gate":
                gain = _innovation_gain(residual, constants)
            estimate = float(wrap_degrees(predicted + gain * residual))
        else:
            estimate = predicted
        output.append(estimate)
    return tuple(output)


def score(estimates, truth, constants=Constants()):
    errors = np.asarray(circular_error(estimates, truth), dtype=float)
    evaluated = errors[constants.warmup_steps:]
    return {
        "mae_degrees": float(np.mean(np.abs(evaluated))),
        "rmse_degrees": float(np.sqrt(np.mean(evaluated ** 2))),
        "within_15deg_fraction": float(np.mean(np.abs(evaluated) <= 15.0)),
        "max_abs_error_degrees": float(np.max(np.abs(evaluated))),
    }


def run_one(scenario, seed, constants=Constants()):
    trial = generate_trial(scenario, seed, constants)
    return {
        "scenario": scenario,
        "seed": seed,
        "algorithms": {
            name: score(estimate_headings(name, trial.sensors, constants), trial.truth, constants)
            for name in ALGORITHMS
        },
    }


def _summary(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(np.mean(values)), "sd": float(np.std(values, ddof=1))}


def _bootstrap_delta(values, seed, samples=10_000):
    """Paired mean delta and percentile CI; negative favors innovation gate."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95": [float(x) for x in np.quantile(means, [0.025, 0.975])],
        "bootstrap_samples": samples,
    }


def build_report(constants=Constants(), seeds=SEEDS):
    runs = [run_one(scenario, seed, constants) for scenario in SCENARIOS for seed in seeds]
    aggregate = {}
    paired = {}
    for scenario_index, scenario in enumerate(SCENARIOS):
        selected = [r for r in runs if r["scenario"] == scenario]
        aggregate[scenario] = {
            algorithm: {
                metric: _summary([r["algorithms"][algorithm][metric] for r in selected])
                for metric in next(iter(r["algorithms"][algorithm] for r in selected))
            }
            for algorithm in ALGORITHMS
        }
        paired[scenario] = {}
        for baseline_index, baseline in enumerate(("fixed_trust", "oracle_reliability", "no_vision")):
            deltas = [r["algorithms"]["innovation_gate"]["mae_degrees"] - r["algorithms"][baseline]["mae_degrees"] for r in selected]
            paired[scenario][f"innovation_gate_minus_{baseline}_mae_degrees"] = _bootstrap_delta(
                deltas, 7100 + 10 * scenario_index + baseline_index
            )
    # The same seed supplies all six scenarios, so the overall interval uses
    # seed as the independent resampling unit rather than treating 120 rows as
    # independent observations.
    all_fixed_deltas = [
        float(np.mean([
            r["algorithms"]["innovation_gate"]["mae_degrees"]
            - r["algorithms"]["fixed_trust"]["mae_degrees"]
            for r in runs if r["seed"] == seed
        ]))
        for seed in seeds
    ]
    paired["all_scenarios"] = {
        "innovation_gate_minus_fixed_trust_mae_degrees": _bootstrap_delta(all_fixed_deltas, 7999)
    }
    return {
        "status": "engineered circular sensor-fusion experiment; not a biological implementation",
        "truth_boundary": "true heading is used only by score(); estimate_headings accepts SensorInputs containing visual, odometry, and the explicitly labeled oracle reliability control",
        "protocol": "constants and seeds frozen before the 20-seed evaluation; algorithms share each generated sensor/disturbance stream",
        "algorithm_notes": {
            "innovation_gate": "fixed hand-selected innovation thresholds; heuristic, not learned reliability",
            "oracle_reliability": "positive control supplied with scenario reliability labels unavailable to the non-oracle methods",
            "no_vision": "initialized from the first visual observation, then integrates odometry without further vision; this calibration favors it over a never-calibrated odometry-only system",
        },
        "bootstrap_units": {
            "per_scenario": "20 paired seeds",
            "all_scenarios": "20 paired seed means across six scenarios (cluster bootstrap by seed)",
        },
        "seeds": list(seeds), "steps": STEPS, "constants": asdict(constants),
        "algorithms": list(ALGORITHMS), "scenarios": list(SCENARIOS),
        "runs": runs, "aggregate": aggregate, "paired_bootstrap_deltas": paired,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/function-bridge-deep/reliability.json")
    args = parser.parse_args()
    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    compact = {scenario: {name: round(data["mae_degrees"]["mean"], 2) for name, data in algorithms.items()} for scenario, algorithms in report["aggregate"].items()}
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
