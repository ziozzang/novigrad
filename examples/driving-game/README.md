# A small driving game that learns from reward

This example connects the existing Novigrad engine to [Farama HighwayEnv](https://highway-env.farama.org/), an open-source driving decision simulator. It is a closed loop: an action changes vehicle dynamics, which change the next observation and score. No prerecorded expert steering labels are used.

The first version uses five vehicles' normalized kinematic observations, not camera pixels. A fixed signed-rate adapter maps these values to the same 319 circuit inputs. Five output meanings are left lane, idle, right lane, faster, and slower. HighwayEnv implements the low-level lane/speed controller; the learned engine chooses these high-level actions.

“Discovering the goal” here means discovering actions that improve an externally supplied scalar game reward through exploration. It does not mean inferring an arbitrary game's rules or inventing a useful objective without any feedback. The environment designer still defines what receives reward. Future intrinsic novelty rewards would also be engineered objectives, not automatic understanding of the game's purpose.

## Mechanism

The policy samples an action from its own probabilities, steps the simulator, and records the observation, action, and scalar reward. Four episodes run at frozen weights. Afterward, discounted reward-to-go minus a lagged time-indexed reward baseline provides a signed learning signal for each chosen action. One mean-gradient update uses the existing `/learn` API. This is episodic REINFORCE with a state-independent variance-reduction baseline. It uses the same reward-gradient core as the earlier dopamine-inspired examples, but the Python runner computes multi-step returns; it does not call the Rust `DelayedReward` wrapper or model biological dopamine dynamics.

Initial models require no labels:

```sh
cargo build --release --features api --bin novi_engine --bin novi_api
./target/release/novi_engine init data/pn_kc.tsv data/kc_mbon.tsv initial.safetensors
```

`init` defaults to five actions, learning rate 0.01, gain 6, active fraction 0.2, homeostasis off and opponent output gains. It refuses to overwrite an existing checkpoint.

## Run on a Mac

After preparing the FlyWire circuit from the main README:

```sh
.venv/bin/python -m pip install -r requirements-simulation.txt
cargo build --release --features api --bin novi_engine --bin novi_api
.venv/bin/python scripts/train_driving_game.py --out results/driving-game-reproduction
```

The script starts its own loopback API processes and closes them on completion. It trains two fixed learning rates and an unrelated-feedback control, uses validation seeds to choose checkpoints, then evaluates twenty held-out environment seeds. It records returns, collision rates, distance, speed, action counts, Safetensors checkpoints, and a GIF from the first held-out episode. Use an empty output directory. Rendering is offscreen; no camera or physical vehicle is accessed.

## Reward and limitations

The configured environment uses collision weight −1, speed weight 0.4, and no right-lane reward. HighwayEnv normalizes the reward. On the road, it returns `(1 − crashed + 0.4 × speed_score) / 1.4`, where `speed_score=clip((forward_speed−20)/10,0,1)`. Thus a slow noncrashed vehicle still receives about 0.714 per step, and a crash ends the episode. The policy may learn to slow down for survival. That must be assessed with distance and speed alongside total return; survival alone does not demonstrate competent driving.

The fast simulator omits collision checks between uncontrolled vehicles. Initial traffic, three lanes, and the short episode duration are deliberately simple. A kinematic observation is privileged relative to a raw camera and already includes object state estimates. Neither a positive result nor fast simulation establishes real vehicle perception, control safety, or general game understanding.

## Watch a saved policy

```sh
.venv/bin/python scripts/play_driving_game.py \
  results/driving-game-reproduction/model.safetensors --human
# Or render without a game window:
.venv/bin/python scripts/play_driving_game.py \
  results/driving-game-reproduction/model.safetensors --gif /tmp/novi-driving.gif
```

Do not set `SDL_VIDEODRIVER=dummy`: this HighwayEnv version disables drawing under that driver. The normal offscreen `rgb_array` mode works on the tested Mac.

Measured outcome: learned collisions 1/20 versus frozen 15/20, return 27.95 versus 22.20. The policy mainly learned to slow down. An always-slower diagnostic achieved 0/20 collisions and return 28.67, so the learned policy does not surpass that simple rule. See the [full report](../../results/DRIVING_GAME_REPORT.md) and [game GIF](../../results/driving-game/driving.gif).

Independent runs start from the same initial tensors. Each episode resets only the simulator; weights persist during training and are frozen throughout validation/test. Advantages use a lagged time-indexed baseline, are divided by 10 and clipped to [-1,1]. These choices and the exact seed separation are recorded in the report.
