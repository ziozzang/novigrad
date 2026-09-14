//! Allocation-free wrapper for one rate-neuron inference/feedback tick.
//! Timing is observed, not guaranteed: this is a soft real-time API.
use crate::plastic::Engine;
use std::path::Path;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug)]
pub struct TickReport {
    pub tick: u64,
    pub compute_time: Duration,
    pub compute_deadline_missed: bool,
    pub selected_action: usize,
}

pub struct Runtime {
    engine: Engine,
    period: Duration,
    output: Vec<f32>,
    ticks: u64,
    compute_deadline_misses: u64,
}

impl Runtime {
    pub fn new(engine: Engine, period: Duration) -> Result<Self, String> {
        if period.is_zero() {
            return Err("period must be positive".into());
        }
        if engine.has_pending_work() {
            return Err("runtime requires an engine without pending trial or batch".into());
        }
        let output = vec![0.0; engine.config().actions];
        Ok(Self {
            engine,
            period,
            output,
            ticks: 0,
            compute_deadline_misses: 0,
        })
    }

    pub fn input_ids(&self) -> &[u64] {
        self.engine.input_ids()
    }
    pub fn output(&self) -> &[f32] {
        &self.output
    }
    pub fn engine(&self) -> &Engine {
        &self.engine
    }
    pub fn period(&self) -> Duration {
        self.period
    }
    pub fn ticks(&self) -> u64 {
        self.ticks
    }
    pub fn compute_deadline_misses(&self) -> u64 {
        self.compute_deadline_misses
    }
    pub fn save_checkpoint<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        self.engine.save_checkpoint(path)
    }

    /// Run one fixed-port trial. Feedback, when present, is applied to this trial.
    /// All caller-controlled values are checked before the engine state changes.
    pub fn tick(
        &mut self,
        input: &[f32],
        feedback: Option<(usize, f32)>,
    ) -> Result<TickReport, String> {
        let start = Instant::now();
        if input.len() != self.engine.input_ids().len()
            || input.iter().any(|&v| !v.is_finite() || v < 0.0)
        {
            return Err("input must be finite nonnegative rates in checkpoint port order".into());
        }
        if let Some((action, reward)) = feedback {
            if action >= self.engine.config().actions || !reward.is_finite() {
                return Err("invalid feedback action or reward".into());
            }
        }
        if self.ticks == u64::MAX {
            return Err("tick counter overflow".into());
        }
        let probabilities = self.engine.forward(input);
        self.output.copy_from_slice(probabilities);
        let selected_action = self
            .output
            .iter()
            .enumerate()
            .max_by(|(ia, a), (ib, b)| a.total_cmp(b).then_with(|| ib.cmp(ia)))
            .map(|(i, _)| i)
            .unwrap();
        if let Some((action, reward)) = feedback {
            self.engine.reward(action, reward)?;
        } else {
            self.engine.clear_pending();
        }
        let compute_time = start.elapsed();
        let compute_deadline_missed = compute_time > self.period;
        self.ticks += 1;
        self.compute_deadline_misses += u64::from(compute_deadline_missed);
        Ok(TickReport {
            tick: self.ticks,
            compute_time,
            compute_deadline_missed,
            selected_action,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plastic::PlasticConfig;
    use std::fs;
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    fn engine_fixture() -> Engine {
        let dir = std::env::temp_dir().join(format!(
            "novi_runtime_{}_{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&dir).unwrap();
        let input = dir.join("input.tsv");
        let plastic = dir.join("plastic.tsv");
        fs::write(&input, "1 10 1 1\n2 20 1 1\n").unwrap();
        fs::write(&plastic, "10 100 1 1\n20 100 1 1\n10 200 1 1\n20 200 1 1\n").unwrap();
        let engine = Engine::load(
            &input,
            &plastic,
            PlasticConfig {
                actions: 2,
                learning_rate: 0.1,
                active_fraction: 1.0,
                homeostasis: false,
                ..Default::default()
            },
        )
        .unwrap();
        fs::remove_dir_all(dir).unwrap();
        engine
    }
    fn fixture() -> Runtime {
        Runtime::new(engine_fixture(), Duration::from_millis(10)).unwrap()
    }
    #[test]
    fn constructor_rejects_pending_trial_and_batch() {
        let mut trial = engine_fixture();
        trial.forward(&[1.0, 0.0]);
        assert!(Runtime::new(trial, Duration::from_millis(10)).is_err());
        let mut batch = engine_fixture();
        batch.forward(&[1.0, 0.0]);
        batch.accumulate_reward(0, 1.0).unwrap();
        assert!(Runtime::new(batch, Duration::from_millis(10)).is_err());
    }
    #[test]
    fn invalid_values_are_atomic_and_unlabeled_ticks_do_not_learn() {
        let mut r = fixture();
        let original = r.engine().weights().collect::<Vec<_>>();
        assert!(r.tick(&[1.0], None).is_err());
        assert!(r.tick(&[f32::NAN, 0.0], None).is_err());
        assert!(r.tick(&[1.0, 0.0], Some((2, 1.0))).is_err());
        assert!(r.tick(&[1.0, 0.0], Some((0, f32::INFINITY))).is_err());
        assert_eq!(r.ticks(), 0);
        assert_eq!(r.engine().weights().collect::<Vec<_>>(), original);
        let first = r.tick(&[1.0, 0.0], None).unwrap();
        let probabilities = r.output().to_vec();
        for _ in 0..5 {
            r.tick(&[1.0, 0.0], None).unwrap();
            assert_eq!(r.output(), probabilities);
        }
        assert_eq!(first.selected_action, 0);
        assert_eq!(r.engine().weights().collect::<Vec<_>>(), original);
    }
    #[test]
    fn repeated_feedback_updates_existing_synapses() {
        let mut r = fixture();
        let original = r.engine().weights().collect::<Vec<_>>();
        for _ in 0..30 {
            r.tick(&[1.0, 0.0], Some((0, 1.0))).unwrap();
        }
        assert_eq!(r.ticks(), 30);
        assert_ne!(r.engine().weights().collect::<Vec<_>>(), original);
        let path =
            std::env::temp_dir().join(format!("novi_runtime_{}.safetensors", std::process::id()));
        r.save_checkpoint(&path).unwrap();
        let loaded = Engine::load_checkpoint(&path).unwrap();
        assert_eq!(
            loaded.weights().collect::<Vec<_>>(),
            r.engine().weights().collect::<Vec<_>>()
        );
        fs::remove_file(path).unwrap();
    }
}
