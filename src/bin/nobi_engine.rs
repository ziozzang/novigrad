//! Data-driven command-line interface to the plastic rate-neuron engine.
use nobi::plastic::{Engine, PlasticConfig};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

const USAGE: &str = "Usage:
  nobi_engine train INPUT_EDGES PLASTIC_EDGES SAMPLES CHECKPOINT [actions] [epochs] [seed]
  nobi_engine eval CHECKPOINT SAMPLES
  nobi_engine infer CHECKPOINT INPUTS
  nobi_engine bench CHECKPOINT [iterations]

Edges: pre<TAB>post<TAB>count<TAB>sign (-1, 0, or 1).
Training/evaluation samples: label<TAB>root_id:value[<TAB>root_id:value...].
Inference INPUTS: one row of root_id:value fields, with no label.
Labels are zero-based action indices; values must be finite and nonnegative.
Output cells, sorted by ID, map to action index modulo actions.
Defaults: actions=2, epochs=10, seed=1. Training metrics use training samples only.
Use separate samples with eval to measure generalization.";

#[derive(Clone)]
struct Sample {
    label: Option<usize>,
    input: Vec<(usize, f32)>,
}

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next(&mut self) -> u64 {
        // SplitMix64 has a defined sequence across platforms, including seed zero.
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    fn uniform(&mut self) -> f64 {
        (self.next() >> 11) as f64 * (1.0 / 9007199254740992.0)
    }
    fn index(&mut self, bound: usize) -> usize {
        (self.next() % bound as u64) as usize
    }
}

fn read_samples(path: &Path, engine: &Engine, labeled: bool) -> Result<Vec<Sample>, String> {
    let ids: BTreeMap<u64, usize> = engine
        .input_ids()
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();
    let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut samples = Vec::new();
    for (line_no, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| format!("{}:{}: {e}", path.display(), line_no + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<_> = line.split('\t').collect();
        let parsed = (|| -> Result<Sample, String> {
            let (label, fields) = if labeled {
                let label: usize = fields[0].parse().map_err(|_| "invalid action label")?;
                if label >= engine.config().actions {
                    return Err("action label out of range".into());
                }
                (Some(label), &fields[1..])
            } else {
                (None, &fields[..])
            };
            if fields.is_empty() {
                return Err("sample must contain at least one root_id:value".into());
            }
            let mut seen = BTreeSet::new();
            let mut input = Vec::with_capacity(fields.len());
            for field in fields {
                let (root, value) = field.split_once(':').ok_or("expected root_id:value")?;
                let root: u64 = root.parse().map_err(|_| "invalid root ID")?;
                let index = *ids.get(&root).ok_or("root ID absent from input edges")?;
                let value: f32 = value.parse().map_err(|_| "invalid input value")?;
                if !value.is_finite() || value < 0.0 {
                    return Err("input value must be finite and nonnegative".into());
                }
                if !seen.insert(index) {
                    return Err("duplicate root ID in sample".into());
                }
                input.push((index, value));
            }
            Ok(Sample { label, input })
        })()
        .map_err(|e| format!("{}:{}: {e}", path.display(), line_no + 1))?;
        samples.push(parsed);
    }
    if samples.is_empty() {
        return Err(format!("{}: no samples", path.display()));
    }
    if !labeled && samples.len() != 1 {
        return Err("inference requires exactly one input row".into());
    }
    Ok(samples)
}

fn forward<'a>(engine: &'a mut Engine, sample: &Sample, scratch: &mut [f32]) -> &'a [f32] {
    scratch.fill(0.0);
    for &(index, value) in &sample.input {
        scratch[index] = value;
    }
    engine.forward(scratch)
}

fn accuracy(engine: &mut Engine, samples: &[Sample], scratch: &mut [f32]) -> (f64, f64) {
    let mut greedy = 0usize;
    let mut expected = 0.0f64;
    for sample in samples {
        let p = forward(engine, sample, scratch);
        let label = sample.label.expect("labeled samples");
        let winner = p
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
            .unwrap()
            .0;
        greedy += usize::from(winner == label);
        expected += p[label] as f64;
        engine.clear_pending();
    }
    (
        greedy as f64 / samples.len() as f64,
        expected / samples.len() as f64,
    )
}

fn train(args: &[String]) -> Result<(), String> {
    if !(6..=9).contains(&args.len()) {
        return Err(USAGE.into());
    }
    let actions = args
        .get(6)
        .map(|s| s.parse::<usize>())
        .transpose()
        .map_err(|_| "invalid actions")?
        .unwrap_or(2);
    let epochs = args
        .get(7)
        .map(|s| s.parse::<usize>())
        .transpose()
        .map_err(|_| "invalid epochs")?
        .unwrap_or(10);
    let seed = args
        .get(8)
        .map(|s| s.parse::<u64>())
        .transpose()
        .map_err(|_| "invalid seed")?
        .unwrap_or(1);
    if epochs == 0 {
        return Err("epochs must be positive".into());
    }
    let mut engine = Engine::load(
        &args[2],
        &args[3],
        PlasticConfig {
            actions,
            ..PlasticConfig::default()
        },
    )?;
    let samples = read_samples(Path::new(&args[4]), &engine, true)?;
    let mut scratch = vec![0.0; engine.input_ids().len()];
    let before = accuracy(&mut engine, &samples, &mut scratch);
    let mut rng = Rng::new(seed);
    let mut order: Vec<_> = (0..samples.len()).collect();
    for _ in 0..epochs {
        for i in (1..order.len()).rev() {
            let j = rng.index(i + 1);
            order.swap(i, j);
        }
        for &i in &order {
            let sample = &samples[i];
            let p = forward(&mut engine, sample, &mut scratch);
            let draw = rng.uniform();
            let mut cumulative = 0.0f64;
            let mut action = p.len() - 1;
            for (j, &probability) in p.iter().enumerate() {
                cumulative += probability as f64;
                if draw < cumulative {
                    action = j;
                    break;
                }
            }
            let reward = if Some(action) == sample.label {
                1.0
            } else {
                -1.0
            };
            engine.reward(action, reward)?;
        }
    }
    let after = accuracy(&mut engine, &samples, &mut scratch);
    engine.save_checkpoint(&args[5])?;
    println!(
        "training samples={} epochs={} seed={} actions={}",
        samples.len(),
        epochs,
        seed,
        actions
    );
    println!(
        "training_before greedy={:.4} expected={:.4}",
        before.0, before.1
    );
    println!(
        "training_after  greedy={:.4} expected={:.4}",
        after.0, after.1
    );
    println!("checkpoint={}", args[5]);
    Ok(())
}

fn eval(args: &[String]) -> Result<(), String> {
    if args.len() != 4 {
        return Err(USAGE.into());
    }
    let mut engine = Engine::load_checkpoint(&args[2])?;
    let samples = read_samples(Path::new(&args[3]), &engine, true)?;
    let mut scratch = vec![0.0; engine.input_ids().len()];
    let (greedy, expected) = accuracy(&mut engine, &samples, &mut scratch);
    println!(
        "evaluation samples={} greedy={:.4} expected={:.4}",
        samples.len(),
        greedy,
        expected
    );
    Ok(())
}

fn infer(args: &[String]) -> Result<(), String> {
    if args.len() != 4 {
        return Err(USAGE.into());
    }
    let mut engine = Engine::load_checkpoint(&args[2])?;
    let samples = read_samples(Path::new(&args[3]), &engine, false)?;
    let mut scratch = vec![0.0; engine.input_ids().len()];
    let p = forward(&mut engine, &samples[0], &mut scratch);
    for (action, probability) in p.iter().enumerate() {
        println!("{action}\t{probability:.6}");
    }
    Ok(())
}

fn bench(args: &[String]) -> Result<(), String> {
    if !(3..=4).contains(&args.len()) {
        return Err(USAGE.into());
    }
    let iterations = args
        .get(3)
        .map(|s| s.parse::<usize>())
        .transpose()
        .map_err(|_| "invalid iterations")?
        .unwrap_or(10_000);
    if iterations == 0 {
        return Err("iterations must be positive".into());
    }
    let load_start = std::time::Instant::now();
    let mut engine = Engine::load_checkpoint(&args[2])?;
    let load = load_start.elapsed();
    let input: Vec<f32> = engine
        .input_ids()
        .iter()
        .map(|&id| if id % 3 == 0 { 0.0 } else { 1.0 })
        .collect();
    for _ in 0..100 {
        std::hint::black_box(engine.forward(&input));
        engine.reward(0, 0.0)?;
    }
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        std::hint::black_box(engine.forward(&input));
        engine.reward(0, 0.0)?;
    }
    let elapsed = start.elapsed();
    println!("load_ms={:.3} input_neurons={} hidden_neurons={} output_neurons={} input_edges={} plastic_edges={}",
        load.as_secs_f64() * 1000.0, engine.input_ids().len(), engine.hidden_ids().len(), engine.output_ids().len(), engine.input_edge_count(), engine.plastic_edge_count());
    println!(
        "iterations={} elapsed_ms={:.3} episodes_per_second={:.1}",
        iterations,
        elapsed.as_secs_f64() * 1000.0,
        iterations as f64 / elapsed.as_secs_f64()
    );
    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let result = match args.get(1).map(String::as_str) {
        Some("train") => train(&args),
        Some("eval") => eval(&args),
        Some("infer") => infer(&args),
        Some("bench") => bench(&args),
        _ => Err(USAGE.into()),
    };
    if let Err(error) = result {
        eprintln!("{error}");
        std::process::exit(2);
    }
}
