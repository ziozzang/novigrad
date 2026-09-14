//! Engineered delayed reward modulation for an unchanged topology-defined engine.
//! This is eligibility replay, not a simulation of dopamine neurons.
use crate::plastic::Engine;
use std::path::Path;

struct TraceEntry {
    input: Vec<f32>,
    action: usize,
}

pub struct DelayedReward {
    engine: Engine,
    trace: Vec<TraceEntry>,
    pending_input: Vec<f32>,
    probabilities: Vec<f32>,
    head: usize,
    count: usize,
    pending: bool,
    lambda: f32,
    baseline_alpha: f32,
    baseline: f32,
    signals: u64,
}

impl DelayedReward {
    pub fn new(
        engine: Engine,
        capacity: usize,
        lambda: f32,
        baseline_alpha: f32,
    ) -> Result<Self, String> {
        if engine.has_pending_work() {
            return Err("controller requires a completed engine trial and batch".into());
        }
        if capacity == 0 || capacity > i32::MAX as usize {
            return Err("trace capacity must be in 1..=i32::MAX".into());
        }
        if !lambda.is_finite()
            || !(0.0..=1.0).contains(&lambda)
            || !baseline_alpha.is_finite()
            || !(0.0..=1.0).contains(&baseline_alpha)
        {
            return Err("lambda and baseline alpha must be finite in [0,1]".into());
        }
        let dimension = engine.input_ids().len();
        let actions = engine.config().actions;
        let trace = (0..capacity)
            .map(|_| TraceEntry {
                input: vec![0.0; dimension],
                action: 0,
            })
            .collect();
        Ok(Self {
            engine,
            trace,
            pending_input: vec![0.0; dimension],
            probabilities: vec![0.0; actions],
            head: 0,
            count: 0,
            pending: false,
            lambda,
            baseline_alpha,
            baseline: 0.0,
            signals: 0,
        })
    }

    pub fn input_ids(&self) -> &[u64] {
        self.engine.input_ids()
    }
    pub fn probabilities(&self) -> &[f32] {
        &self.probabilities
    }
    pub fn baseline(&self) -> f32 {
        self.baseline
    }
    pub fn trace_len(&self) -> usize {
        self.count
    }
    pub fn trace_capacity(&self) -> usize {
        self.trace.len()
    }
    pub fn signals(&self) -> u64 {
        self.signals
    }
    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    /// Read-only policy evaluation between episodes; it never records eligibility.
    pub fn evaluate(&mut self, input: &[f32]) -> Result<&[f32], String> {
        if self.pending || self.count > 0 {
            return Err("evaluation requires no open episode".into());
        }
        if input.len() != self.pending_input.len()
            || input.iter().any(|&x| !x.is_finite() || x < 0.0)
        {
            return Err("input must be finite nonnegative rates in checkpoint port order".into());
        }
        self.probabilities
            .copy_from_slice(self.engine.forward(input));
        self.engine.clear_pending();
        Ok(&self.probabilities)
    }

    /// Observe a fixed-port rate vector, then call record_action exactly once.
    pub fn observe(&mut self, input: &[f32]) -> Result<&[f32], String> {
        if self.pending || self.count == self.trace.len() {
            return Err(
                "record the current action or signal reward before another observation".into(),
            );
        }
        if input.len() != self.pending_input.len()
            || input.iter().any(|&x| !x.is_finite() || x < 0.0)
        {
            return Err("input must be finite nonnegative rates in checkpoint port order".into());
        }
        self.pending_input.copy_from_slice(input);
        self.probabilities
            .copy_from_slice(self.engine.forward(input));
        self.pending = true;
        Ok(&self.probabilities)
    }

    /// Store the chosen action without updating synapses. Full traces are rejected.
    pub fn record_action(&mut self, action: usize) -> Result<(), String> {
        if !self.pending {
            return Err("record_action requires an observation".into());
        }
        if action >= self.engine.config().actions {
            return Err("action out of range".into());
        }
        let slot = (self.head + self.count) % self.trace.len();
        self.trace[slot].input.copy_from_slice(&self.pending_input);
        self.trace[slot].action = action;
        self.count += 1;
        self.engine.clear_pending();
        self.pending = false;
        Ok(())
    }

    /// Apply one mean gradient from the buffered episode using a reward prediction error.
    /// Newest trace has age zero; older traces receive `lambda.powi(age)`.
    pub fn signal(&mut self, reward: f32) -> Result<(), String> {
        if self.pending || self.count == 0 {
            return Err("signal requires a completed nonempty trace".into());
        }
        if !reward.is_finite() || !(-1.0..=1.0).contains(&reward) {
            return Err("reward must be finite in [-1,1]".into());
        }
        let error = reward - self.baseline;
        for i in 0..self.count {
            let slot = (self.head + i) % self.trace.len();
            let item = &self.trace[slot];
            self.engine.forward(&item.input);
            let age = (self.count - 1 - i) as i32;
            self.engine
                .accumulate_reward(item.action, error * self.lambda.powi(age))?;
        }
        self.engine.apply_batch()?;
        self.baseline += self.baseline_alpha * error;
        self.head = (self.head + self.count) % self.trace.len();
        self.count = 0;
        self.signals += 1;
        Ok(())
    }

    pub fn save_checkpoint<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        if self.pending || self.count > 0 {
            return Err("checkpoint requires a signaled or discarded episode".into());
        }
        self.engine.save_checkpoint(path)
    }

    /// Drop the current episode without changing weights or baseline.
    pub fn discard_episode(&mut self) {
        self.engine.discard_batch();
        self.pending = false;
        self.head = (self.head + self.count) % self.trace.len();
        self.count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plastic::PlasticConfig;
    use std::fs;
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    fn fixture() -> DelayedReward {
        let dir = std::env::temp_dir().join(format!(
            "novi_mod_{}_{}",
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
        DelayedReward::new(engine, 2, 0.8, 0.5).unwrap()
    }
    #[test]
    fn no_learning_before_delayed_signal_and_capacity_is_bounded() {
        let mut m = fixture();
        let initial = m.engine().weights().collect::<Vec<_>>();
        m.observe(&[1.0, 0.0]).unwrap();
        assert!(m.observe(&[1.0, 0.0]).is_err());
        assert!(m.record_action(2).is_err());
        m.record_action(0).unwrap();
        m.observe(&[1.0, 0.0]).unwrap();
        m.record_action(0).unwrap();
        assert!(m.observe(&[1.0, 0.0]).is_err());
        assert!(m
            .save_checkpoint("/tmp/novi_unfinished.safetensors")
            .is_err());
        assert_eq!(m.engine().weights().collect::<Vec<_>>(), initial);
        m.signal(1.0).unwrap();
        assert_eq!(m.trace_len(), 0);
        assert_eq!(m.baseline(), 0.5);
        assert_ne!(m.engine().weights().collect::<Vec<_>>(), initial);
    }
    #[test]
    fn reward_error_sign_and_invalid_signal_are_atomic() {
        let mut positive = fixture();
        let mut negative = fixture();
        let initial = positive.engine().weights().collect::<Vec<_>>();
        for m in [&mut positive, &mut negative] {
            m.observe(&[1.0, 0.0]).unwrap();
            m.record_action(0).unwrap();
        }
        assert!(positive.signal(f32::NAN).is_err());
        assert_eq!(positive.trace_len(), 1);
        assert_eq!(positive.baseline(), 0.0);
        assert_eq!(positive.engine().weights().collect::<Vec<_>>(), initial);
        positive.signal(1.0).unwrap();
        negative.signal(-1.0).unwrap();
        let p = positive.engine().weights().next().unwrap();
        let n = negative.engine().weights().next().unwrap();
        assert!(p > initial[0] && n < initial[0]);
        assert_eq!(positive.baseline(), 0.5);
        assert_eq!(negative.baseline(), -0.5);
    }
}
