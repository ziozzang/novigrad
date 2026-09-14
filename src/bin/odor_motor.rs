//! Synthetic odor-to-virtual-action delayed reward example on the real PN-KC-MBON graph.
use novi::modulation::DelayedReward;
use novi::plastic::{Engine, PlasticConfig};
use std::env;
use std::error::Error;
use std::path::PathBuf;

type R<T> = Result<T, Box<dyn Error>>;

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
        (self.next() >> 40) as f32 / 16_777_216.0
    }
}

fn odor(dimension: usize, label: usize, rng: &mut Rng, rates: &mut [f32]) {
    assert_eq!(dimension, rates.len());
    rates.fill(0.0);
    // Two disjoint groups of 16 real ALPN input ports, varied trial by trial.
    for _ in 0..12 {
        rates[label * 16 + (rng.next() % 16) as usize] = 1.0;
    }
    for _ in 0..4 {
        rates[(1 - label) * 16 + (rng.next() % 16) as usize] = 0.15;
    }
}

fn accuracy(controller: &mut DelayedReward, seed: u64, samples: usize, reversed: bool) -> R<f32> {
    let mut rng = Rng::new(seed);
    let mut rates = vec![0.0; controller.input_ids().len()];
    let mut correct = 0usize;
    for i in 0..samples {
        let label = i % 2;
        odor(rates.len(), label, &mut rng, &mut rates);
        let p = controller.evaluate(&rates)?;
        let predicted = usize::from(p[1] > p[0]);
        correct += usize::from(predicted == if reversed { 1 - label } else { label });
    }
    Ok(correct as f32 / samples as f32)
}

fn run() -> R<()> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 11
        || args[1] != "--episodes"
        || args[3] != "--seed"
        || args[5] != "--mode"
        || args[7] != "--reverse-after"
        || args[9] != "--out"
    {
        return Err("usage: odor_motor --episodes N --seed N --mode reward|shuffled|frozen --reverse-after N --out MODEL.safetensors".into());
    }
    let episodes: usize = args[2].parse()?;
    let seed: u64 = args[4].parse()?;
    let mode = args[6].as_str();
    let reverse_after: usize = args[8].parse()?;
    let output = PathBuf::from(&args[10]);
    if episodes == 0 {
        return Err("episodes must be positive".into());
    }
    if !matches!(mode, "reward" | "shuffled" | "frozen") || reverse_after > episodes {
        return Err("invalid mode or reverse-after".into());
    }
    let config = PlasticConfig {
        actions: 2,
        learning_rate: 0.02,
        logit_gain: 6.0,
        active_fraction: 0.1,
        ..Default::default()
    };
    let engine = Engine::load("data/pn_kc.tsv", "data/kc_mbon.tsv", config)?;
    if engine.input_ids().len() < 32 {
        return Err("need at least 32 PN ports".into());
    }
    let mut controller = DelayedReward::new(engine, 3, 0.8, 0.02)?;
    let initial_weights = controller.engine().weights().collect::<Vec<_>>();
    let frozen_accuracy = accuracy(&mut controller, 0xdead_beef, 400, false)?;
    let mut pre_reversal_accuracy = None;
    let mut rng = Rng::new(seed);
    let mut rates = vec![0.0; controller.input_ids().len()];
    for episode in 0..episodes {
        if episode == reverse_after && reverse_after < episodes {
            pre_reversal_accuracy = Some(accuracy(&mut controller, 0xdead_beef, 400, false)?);
        }
        let label = episode % 2;
        odor(rates.len(), label, &mut rng, &mut rates);
        let p = controller.observe(&rates)?;
        let action = usize::from(rng.unit() >= p[0]);
        controller.record_action(action)?;
        // Two more observations occur before the same episode-level outcome arrives.
        for _ in 0..2 {
            controller.observe(&rates)?;
            controller.record_action(action)?;
        }
        if mode == "frozen" {
            controller.discard_episode();
        } else {
            let target = if mode == "shuffled" {
                (rng.next() & 1) as usize
            } else if episode >= reverse_after {
                1 - label
            } else {
                label
            };
            controller.signal(if action == target { 1.0 } else { -1.0 })?;
        }
    }
    let trained_accuracy = accuracy(
        &mut controller,
        0xdead_beef,
        400,
        reverse_after < episodes && mode == "reward",
    )?;
    let changed = controller
        .engine()
        .weights()
        .zip(initial_weights)
        .filter(|(w, initial)| w != initial)
        .count();
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    controller.save_checkpoint(&output)?;
    let pre = pre_reversal_accuracy
        .map(|x| format!("{x:.6}"))
        .unwrap_or("null".into());
    println!("{{\"example\":\"synthetic_odor_virtual_action\",\"mode\":\"{mode}\",\"seed\":{seed},\"episodes\":{episodes},\"reverse_after\":{reverse_after},\"delay_steps\":3,\"frozen_accuracy\":{frozen_accuracy:.6},\"pre_reversal_accuracy\":{pre},\"final_accuracy\":{trained_accuracy:.6},\"baseline\":{:.6},\"changed_synapses\":{changed},\"checkpoint\":\"{}\"}}",controller.baseline(),output.display());
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("odor_motor: {error}");
        std::process::exit(1);
    }
}
