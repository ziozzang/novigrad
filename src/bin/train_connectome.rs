//! Reproducible reward-learning experiments on real anatomical synapses.
use nobi::plastic::{Engine, PlasticConfig};
use std::{
    error::Error,
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Clone, Copy, Debug)]
enum Task {
    Binary,
    FourWay,
    Xor,
}
impl Task {
    fn name(self) -> &'static str {
        match self {
            Self::Binary => "binary",
            Self::FourWay => "fourway",
            Self::Xor => "xor",
        }
    }
    fn cues(self) -> usize {
        match self {
            Self::Binary => 2,
            _ => 4,
        }
    }
    fn actions(self) -> usize {
        match self {
            Self::FourWay => 4,
            _ => 2,
        }
    }
    fn target(self, cue: usize, reverse: bool) -> usize {
        let target = match self {
            Self::Xor => (cue & 1) ^ ((cue >> 1) & 1),
            _ => cue,
        };
        (target + usize::from(reverse)) % self.actions()
    }
}
#[derive(Clone, Copy, Debug)]
enum Mode {
    Reward,
    Frozen,
    Shuffled,
}
impl Mode {
    fn name(self) -> &'static str {
        match self {
            Self::Reward => "reward",
            Self::Frozen => "frozen",
            Self::Shuffled => "shuffled",
        }
    }
}
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32) / 16_777_216.0
    }
    fn sample(&mut self, p: &[f32]) -> usize {
        let mut u = self.unit();
        for (i, value) in p.iter().enumerate() {
            u -= value;
            if u < 0.0 {
                return i;
            }
        }
        p.len() - 1
    }
}
fn hash(mut x: u64) -> u64 {
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}
struct Stimuli {
    groups: Vec<usize>,
    input: Vec<f32>,
    task: Task,
}
impl Stimuli {
    fn new(ids: &[u64], task: Task, seed: u64) -> Self {
        let groups = ids
            .iter()
            .map(|id| (hash(id ^ seed.wrapping_mul(0x9e37_79b9)) % task.cues() as u64) as usize)
            .collect();
        Self {
            groups,
            input: vec![0.0; ids.len()],
            task,
        }
    }
    fn generate(&mut self, cue: usize, rng: &mut Rng) -> &[f32] {
        for (x, &group) in self.input.iter_mut().zip(&self.groups) {
            let on = match self.task {
                Task::Xor => group == (cue & 1) || group == 2 + ((cue >> 1) & 1),
                _ => group == cue,
            };
            // Independent dropout and concentration jitter, with weak background.
            let keep = rng.unit() >= 0.1;
            let amplitude = 0.8 + 0.4 * rng.unit();
            *x = if on && keep {
                amplitude
            } else {
                0.02 * amplitude
            };
        }
        &self.input
    }
}
#[derive(Clone, Copy, Default)]
struct Metric {
    greedy: f64,
    sampled: f64,
    expected: f64,
}
fn evaluate(
    engine: &mut Engine,
    stimuli: &mut Stimuli,
    task: Task,
    reverse: bool,
    seed: u64,
    trials: usize,
) -> Metric {
    // Fresh held-out streams. These never advance training RNGs.
    let mut noise = Rng::new(hash(seed ^ 0xeeee_0123));
    let mut decisions = Rng::new(hash(seed ^ 0xeeee_4567));
    let mut result = Metric::default();
    for trial in 0..trials {
        let cue = trial % task.cues();
        let target = task.target(cue, reverse);
        let p = engine.forward(stimuli.generate(cue, &mut noise));
        let greedy = p
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
            .unwrap()
            .0;
        result.greedy += f64::from(greedy == target);
        result.sampled += f64::from(decisions.sample(p) == target);
        result.expected += f64::from(p[target]);
    }
    result.greedy /= trials as f64;
    result.sampled /= trials as f64;
    result.expected /= trials as f64;
    engine.clear_pending();
    result
}
fn train(
    engine: &mut Engine,
    stimuli: &mut Stimuli,
    task: Task,
    mode: Mode,
    reverse: bool,
    seed: u64,
    episodes: usize,
) -> Result<(), String> {
    let mut environment = Rng::new(hash(seed ^ 0x1234_abcd ^ u64::from(reverse)));
    let mut actions = Rng::new(hash(seed ^ 0x5678_abcd ^ u64::from(reverse)));
    let mut random_reward = Rng::new(hash(seed ^ 0x9999_abcd ^ u64::from(reverse)));
    for _ in 0..episodes {
        let cue = (environment.next() % task.cues() as u64) as usize;
        let p = engine.forward(stimuli.generate(cue, &mut environment));
        let action = actions.sample(p);
        let reward = match mode {
            Mode::Reward => {
                if action == task.target(cue, reverse) {
                    1.0
                } else {
                    -1.0
                }
            }
            Mode::Frozen => 0.0,
            Mode::Shuffled => {
                if random_reward.unit() < 0.5 {
                    1.0
                } else {
                    -1.0
                }
            }
        };
        engine.reward(action, reward)?;
    }
    Ok(())
}
struct ResultRow {
    task: Task,
    mode: Mode,
    seed: u64,
    before: Metric,
    after: Metric,
    before_reversal: Metric,
    reversal: Metric,
    changed: usize,
    l1_delta: f64,
    seconds: f64,
}
#[derive(Clone, Copy)]
struct ExperimentSettings {
    episodes: usize,
    trials: usize,
    learning_rate: f32,
    reversal_episodes: usize,
    seed_start: u64,
    logit_gain: f32,
}
fn run(
    data: &Path,
    out: &Path,
    task: Task,
    mode: Mode,
    seed: u64,
    settings: ExperimentSettings,
) -> Result<ResultRow, Box<dyn Error>> {
    let ExperimentSettings {
        episodes,
        trials,
        learning_rate,
        reversal_episodes,
        seed_start,
        logit_gain,
    } = settings;
    let config = PlasticConfig {
        actions: task.actions(),
        learning_rate,
        logit_gain,
        ..Default::default()
    };
    let mut engine = Engine::load(data.join("pn_kc.tsv"), data.join("kc_mbon.tsv"), config)?;
    let mut stimuli = Stimuli::new(engine.input_ids(), task, seed);
    let initial = engine.weights().collect::<Vec<_>>();
    let before = evaluate(&mut engine, &mut stimuli, task, false, seed, trials);
    assert_eq!(
        engine.weights().collect::<Vec<_>>(),
        initial,
        "evaluation mutated weights"
    );
    let start = Instant::now();
    train(&mut engine, &mut stimuli, task, mode, false, seed, episodes)?;
    let learned = engine.weights().collect::<Vec<_>>();
    let changed = initial
        .iter()
        .zip(&learned)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    let l1_delta = initial
        .iter()
        .zip(&learned)
        .map(|(a, b)| f64::from((a - b).abs()))
        .sum();
    let after = evaluate(&mut engine, &mut stimuli, task, false, seed, trials);
    let before_reversal = evaluate(
        &mut engine,
        &mut stimuli,
        task,
        true,
        seed ^ 0xdead_beef,
        trials,
    );
    assert_eq!(
        engine.weights().collect::<Vec<_>>(),
        learned,
        "evaluation mutated weights"
    );
    if matches!(mode, Mode::Frozen) {
        assert_eq!(changed, 0, "zero reward mutated weights");
    }
    if seed == seed_start && matches!(mode, Mode::Reward) {
        let checkpoint = out.join(format!("{}.safetensors", task.name()));
        engine.save_checkpoint(&checkpoint)?;
        let mut restored = Engine::load_checkpoint(&checkpoint)?;
        assert_eq!(
            restored.weights().collect::<Vec<_>>(),
            learned,
            "checkpoint lost weights"
        );
        let repeated = evaluate(&mut restored, &mut stimuli, task, false, seed, trials);
        assert_eq!(
            repeated.expected.to_bits(),
            after.expected.to_bits(),
            "checkpoint changed predictions"
        );
        // Continuation must be bit-identical with identical external input/action/reward streams.
        train(
            &mut restored,
            &mut stimuli,
            task,
            mode,
            true,
            seed,
            reversal_episodes,
        )?;
        train(
            &mut engine,
            &mut stimuli,
            task,
            mode,
            true,
            seed,
            reversal_episodes,
        )?;
        assert_eq!(
            restored.weights().collect::<Vec<_>>(),
            engine.weights().collect::<Vec<_>>(),
            "checkpoint resume diverged"
        );
        let mut ports = File::create(out.join(format!("{}_ports.tsv", task.name())))?;
        writeln!(ports, "root_id\tinput_group")?;
        for (id, group) in engine.input_ids().iter().zip(&stimuli.groups) {
            writeln!(ports, "{id}\t{group}")?;
        }
    } else {
        train(
            &mut engine,
            &mut stimuli,
            task,
            mode,
            true,
            seed,
            reversal_episodes,
        )?;
    }
    let reversal = evaluate(
        &mut engine,
        &mut stimuli,
        task,
        true,
        seed ^ 0xdead_beef,
        trials,
    );
    Ok(ResultRow {
        task,
        mode,
        seed,
        before,
        after,
        before_reversal,
        reversal,
        changed,
        l1_delta,
        seconds: start.elapsed().as_secs_f64(),
    })
}
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    let allowed = [
        "--data",
        "--out",
        "--episodes",
        "--seeds",
        "--eval",
        "--lr",
        "--reversal-episodes",
        "--task",
        "--seed-start",
        "--gain",
    ];
    if !(args.len() - 1).is_multiple_of(2)
        || args[1..]
            .chunks(2)
            .any(|pair| !allowed.contains(&pair[0].as_str()))
    {
        return Err("usage: train_connectome [--data DIR] [--out DIR] [--episodes N] [--reversal-episodes N] [--seeds N] [--eval N] [--lr RATE] [--gain GAIN] [--seed-start SEED] [--task binary|fourway|xor|all]".into());
    }
    let value = |flag: &str| {
        args.iter()
            .position(|s| s == flag)
            .and_then(|i| args.get(i + 1))
    };
    let data = PathBuf::from(value("--data").map(String::as_str).unwrap_or("data"));
    let out = PathBuf::from(
        value("--out")
            .map(String::as_str)
            .unwrap_or("results/training"),
    );
    let episodes: usize = value("--episodes")
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(2000);
    let seeds: u64 = value("--seeds")
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(8);
    let seed_start: u64 = value("--seed-start")
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(1);
    let seed_end = seed_start.checked_add(seeds).ok_or("seed range overflow")?;
    let trials: usize = value("--eval")
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(512);
    let learning_rate: f32 = value("--lr")
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(0.02);
    let logit_gain: f32 = value("--gain")
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(12.0);
    let reversal_episodes: usize = value("--reversal-episodes")
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(2000);
    if episodes == 0
        || reversal_episodes == 0
        || seeds == 0
        || trials < 4
        || !trials.is_multiple_of(4)
    {
        return Err(
            "episodes/seeds must be positive; eval must be a positive multiple of 4".into(),
        );
    }
    let tasks = match value("--task").map(String::as_str).unwrap_or("all") {
        "binary" => vec![Task::Binary],
        "fourway" => vec![Task::FourWay],
        "xor" => vec![Task::Xor],
        "all" => vec![Task::Binary, Task::FourWay, Task::Xor],
        _ => return Err("task must be binary, fourway, xor, or all".into()),
    };
    fs::create_dir_all(&out)?;
    let mut csv = File::create(out.join("metrics.csv"))?;
    writeln!(csv, "task,mode,seed,episodes,eval_trials,before_greedy,before_expected,after_greedy,after_sampled,after_expected,reversal_greedy,reversal_expected,changed_weights,l1_delta,seconds,before_reversal_expected,reversal_episodes,learning_rate,logit_gain")?;
    let mut rows = Vec::new();
    let settings = ExperimentSettings {
        episodes,
        trials,
        learning_rate,
        reversal_episodes,
        seed_start,
        logit_gain,
    };
    for &task in &tasks {
        for seed in seed_start..seed_end {
            for mode in [Mode::Reward, Mode::Frozen, Mode::Shuffled] {
                let r = run(&data, &out, task, mode, seed, settings)?;
                println!("{} {} seed={seed}: {:.1}% -> {:.1}% (expected {:.1}%), reversal {:.1}%; {} weights changed; {:.2}s", task.name(), mode.name(), r.before.greedy*100.0, r.after.greedy*100.0, r.after.expected*100.0, r.reversal.greedy*100.0, r.changed, r.seconds);
                writeln!(csv, "{},{},{},{episodes},{trials},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8},{},{:.8},{:.6},{:.8},{reversal_episodes},{learning_rate},{logit_gain}", r.task.name(), r.mode.name(), r.seed, r.before.greedy, r.before.expected, r.after.greedy, r.after.sampled, r.after.expected, r.reversal.greedy, r.reversal.expected, r.changed, r.l1_delta, r.seconds, r.before_reversal.expected)?;
                csv.flush()?;
                rows.push(r);
            }
        }
    }
    let mut passed = true;
    let mut report = File::create(out.join("summary.txt"))?;
    writeln!(report, "Topology-constrained rate learning; {episodes} training + {reversal_episodes} reversal episodes per run; lr={learning_rate}, gain={logit_gain}, homeostasis=true; {seeds} seeds starting at {seed_start}; {trials} held-out noisy trials. No anatomy superiority or in-vivo claim.")?;
    for task in tasks {
        let mean = |mode: Mode, reversal: bool| -> f64 {
            let matching: Vec<_> = rows
                .iter()
                .filter(|r| r.task.name() == task.name() && r.mode.name() == mode.name())
                .collect();
            matching
                .iter()
                .map(|r| {
                    if reversal {
                        r.reversal.expected
                    } else {
                        r.after.expected
                    }
                })
                .sum::<f64>()
                / matching.len() as f64
        };
        let trained = mean(Mode::Reward, false);
        let frozen = mean(Mode::Frozen, false);
        let shuffled = mean(Mode::Shuffled, false);
        let reversed = mean(Mode::Reward, true);
        let frozen_reversal = mean(Mode::Frozen, true);
        let random_reversal = mean(Mode::Shuffled, true);
        let all_seeds = rows
            .iter()
            .filter(|r| r.task.name() == task.name() && matches!(r.mode, Mode::Reward))
            .all(|r| r.after.expected >= 0.75 && r.reversal.expected >= 0.75 && r.changed > 0);
        let ok = trained >= 0.85
            && reversed >= 0.85
            && trained - frozen >= 0.20
            && trained - shuffled >= 0.20
            && reversed - frozen_reversal >= 0.20
            && reversed - random_reversal >= 0.20
            && all_seeds;
        let line = format!("{}: reward={trained:.4}, frozen={frozen:.4}, random_reward={shuffled:.4}, reversal={reversed:.4}, reversal_frozen={frozen_reversal:.4}, reversal_random={random_reversal:.4}, pass={ok}", task.name());
        println!("{line}");
        writeln!(report, "{line}")?;
        passed &= ok;
    }
    if !passed {
        return Err("training acceptance criteria failed; inspect metrics.csv".into());
    }
    Ok(())
}
