#!/usr/bin/env python3
"""Engineered circular heading estimator and steering example.

This is a functional software mechanism inspired by navigation experiments. It
does not reproduce, identify, or calibrate any fly neuron or anatomical circuit.
Ground-truth heading is held by the evaluator and is never passed to the
estimator or steering controller.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def wrap_degrees(value):
    """Wrap scalar or array angles to [-180, 180)."""
    return (np.asarray(value) + 180.0) % 360.0 - 180.0


def circular_error(estimate, reference):
    return wrap_degrees(np.asarray(estimate) - np.asarray(reference))


@dataclass(frozen=True)
class NavigationConstants:
    prior_weight: float = 1.0
    visual_weight: float = 4.0
    fixed_visual_reliability: float = 1.0
    odometry_gain: float = 1.0
    steering_gain: float = 0.55
    max_turn_degrees: float = 24.0


class CircularHeadingEstimator:
    """Persistent one-angle state with optional visual and odometry updates."""

    def __init__(self, constants=NavigationConstants(), use_reported_reliability=True):
        self.constants = constants
        self.use_reported_reliability = bool(use_reported_reliability)
        self.reset()

    def reset(self):
        self._heading = None

    @property
    def heading(self):
        return self._heading

    def update(self, visual_heading=None, odometry_delta=None, visual_reliability=1.0):
        for name, value in (("visual_heading", visual_heading), ("odometry_delta", odometry_delta)):
            if value is not None and not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not np.isfinite(visual_reliability) or not 0.0 <= visual_reliability <= 1.0:
            raise ValueError("visual_reliability must be finite and in [0, 1]")

        predicted = self._heading
        if predicted is not None and odometry_delta is not None:
            predicted = float(wrap_degrees(predicted + self.constants.odometry_gain * odometry_delta))
        if visual_heading is None:
            self._heading = predicted
            return self._heading

        visual = float(wrap_degrees(visual_heading))
        if predicted is None:
            self._heading = visual
            return self._heading

        reliability = visual_reliability if self.use_reported_reliability else self.constants.fixed_visual_reliability
        prior_radians = np.deg2rad(predicted)
        visual_radians = np.deg2rad(visual)
        x = self.constants.prior_weight * np.cos(prior_radians)
        y = self.constants.prior_weight * np.sin(prior_radians)
        x += self.constants.visual_weight * reliability * np.cos(visual_radians)
        y += self.constants.visual_weight * reliability * np.sin(visual_radians)
        if np.hypot(x, y) < 1e-12:
            self._heading = predicted
        else:
            self._heading = float(wrap_degrees(np.rad2deg(np.arctan2(y, x))))
        return self._heading


class AngularSteeringComparator:
    def __init__(self, constants=NavigationConstants()):
        self.constants = constants
        self.target_bearing = 0.0

    def set_target(self, bearing):
        if not np.isfinite(bearing):
            raise ValueError("target bearing must be finite")
        self.target_bearing = float(wrap_degrees(bearing))

    def choose_turn(self, estimated_heading):
        if estimated_heading is None or not np.isfinite(estimated_heading):
            raise RuntimeError("heading estimate unavailable")
        error = float(wrap_degrees(self.target_bearing - estimated_heading))
        return float(np.clip(
            self.constants.steering_gain * error,
            -self.constants.max_turn_degrees,
            self.constants.max_turn_degrees,
        ))


def scenario_config(name):
    configs = {
        "visible_cue": {"visual": "normal", "odometry": True, "switch_step": None},
        "cue_dropout_odometry": {"visual": "dropout", "odometry": True, "switch_step": None},
        "cue_dropout_no_odometry": {"visual": "dropout", "odometry": False, "switch_step": None},
        "conflicting_unreliable_cue": {"visual": "conflict", "odometry": True, "switch_step": None},
        "conflicting_fixed_trust": {"visual": "conflict", "odometry": True, "switch_step": None,
                                      "fixed_trust": True},
        "goal_switch": {"visual": "normal", "odometry": True, "switch_step": 60},
    }
    return configs[name]


def run_episode(name, seed, steps=120, constants=NavigationConstants()):
    """Run a deterministic seeded simulation; truth remains evaluator-local."""
    config = scenario_config(name)
    rng = np.random.default_rng(seed)
    estimator = CircularHeadingEstimator(constants, not config.get("fixed_trust", False))
    controller = AngularSteeringComparator(constants)
    controller.set_target(70.0)
    true_heading = -110.0
    analytic_heading = true_heading
    last_motion = None
    records = []

    for step in range(steps):
        if config["switch_step"] == step:
            controller.set_target(-65.0)
        in_dropout = 35 <= step < 75
        visual = float(wrap_degrees(true_heading + rng.normal(0.0, 3.0)))
        reliability = 1.0
        if config["visual"] == "dropout" and in_dropout:
            visual = None
        elif config["visual"] == "conflict" and in_dropout:
            visual = float(wrap_degrees(true_heading + 115.0 + rng.normal(0.0, 3.0)))
            reliability = 0.005

        odometry = None
        if config["odometry"] and last_motion is not None:
            odometry = last_motion + rng.normal(0.0, 0.7)
        estimate = estimator.update(visual, odometry, reliability)
        turn = controller.choose_turn(estimate)
        analytic_turn = float(np.clip(
            constants.steering_gain * float(wrap_degrees(controller.target_bearing - analytic_heading)),
            -constants.max_turn_degrees,
            constants.max_turn_degrees,
        ))

        disturbance = 5.0 * np.sin(step * 0.31) + rng.normal(0.0, 0.8)
        motion = turn + disturbance
        records.append({
            "step": step,
            "cue_available": visual is not None,
            "cue_reliability": reliability if visual is not None else None,
            "odometry_available": odometry is not None,
            "target_bearing": controller.target_bearing,
            "true_heading": true_heading,
            "estimated_heading": estimate,
            "heading_error": float(circular_error(estimate, true_heading)),
            "turn": turn,
            "analytic_true_heading": analytic_heading,
            "analytic_true_heading_turn": analytic_turn,
            "target_error": float(circular_error(true_heading, controller.target_bearing)),
            "analytic_target_error": float(circular_error(analytic_heading, controller.target_bearing)),
            "dropout_window": in_dropout,
        })
        true_heading = float(wrap_degrees(true_heading + motion))
        analytic_heading = float(wrap_degrees(analytic_heading + analytic_turn + disturbance))
        last_motion = motion

    dropout = [row for row in records if row["dropout_window"]]
    evaluation = records[config["switch_step"]:] if config["switch_step"] is not None else records[20:]
    return {
        "name": name,
        "seed": seed,
        "metrics": {
            "dropout_heading_rmse_degrees": float(np.sqrt(np.mean([r["heading_error"] ** 2 for r in dropout]))),
            "target_alignment_mean_abs_error_degrees": float(np.mean([abs(r["target_error"]) for r in evaluation])),
            "target_alignment_within_15deg_fraction": float(np.mean([abs(r["target_error"]) <= 15.0 for r in evaluation])),
            "analytic_control_target_alignment_mean_abs_error_degrees": float(np.mean([
                abs(r["analytic_target_error"]) for r in evaluation
            ])),
            "analytic_control_target_alignment_within_15deg_fraction": float(np.mean([
                abs(r["analytic_target_error"]) <= 15.0 for r in evaluation
            ])),
            "mean_abs_turn_difference_from_true_heading_control": float(np.mean([
                abs(r["turn"] - r["analytic_true_heading_turn"]) for r in records
            ])),
        },
        "records": records,
    }


def build_report(seeds=(17, 29, 43)):
    constants = NavigationConstants()
    names = [
        "visible_cue", "cue_dropout_odometry", "cue_dropout_no_odometry",
        "conflicting_unreliable_cue", "conflicting_fixed_trust", "goal_switch",
    ]
    runs = [run_episode(name, seed, constants=constants) for name in names for seed in seeds]
    aggregate = {}
    for name in names:
        selected = [run["metrics"] for run in runs if run["name"] == name]
        aggregate[name] = {
            key: {"mean": float(np.mean([row[key] for row in selected])),
                  "values_by_seed": [row[key] for row in selected]}
            for key in selected[0]
        }
    return {
        "status": "source-inspired engineered functional example; not an anatomical or neuron-level model",
        "mechanism": "persistent circular heading estimate + optional host visual cue + optional odometry + separate target bearing + wrapped angular proportional steering",
        "truth_boundary": "true_heading is used only by the simulator/evaluator and analytic comparison; it is never passed to CircularHeadingEstimator or AngularSteeringComparator",
        "constants": constants.__dict__,
        "seeds": list(seeds),
        "source_inspiration": [
            "Fly heading work motivates persistence, cue tethering, and odometric updating as test cases.",
            "Goal-heading comparison work motivates separating estimated heading from target bearing.",
            "No claim is made that this implementation reproduces FC2, PFL, EPG, or other neurons.",
        ],
        "aggregate": aggregate,
        "runs": runs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/function-bridge/navigation.json")
    args = parser.parse_args()
    report = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["aggregate"], indent=2))


if __name__ == "__main__":
    main()
