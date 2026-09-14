//! Novigrad library: connectome activation and configurable plastic rate learning.
//! The legacy `Agent` below is a small synthetic reward-learning fixture; use
//! `plastic::Engine` for topology-defined tasks and Safetensors models.

pub mod connectome;
pub mod modulation;
pub mod plastic;
pub mod runtime;

pub const CUES: usize = 2;
pub const ACTIONS: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Config {
    /// Total number of synthetic Kenyon cells. Must be even.
    pub kcs: usize,
    /// Number of active cells per odor. Must be in 1..=kcs/2.
    pub active_kcs: usize,
    /// Reward-modulated policy-gradient step size.
    pub learning_rate: f32,
    /// Absolute synaptic weight limit.
    pub weight_limit: f32,
    /// Softmax temperature for training-time exploration.
    pub temperature: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            kcs: 256,
            active_kcs: 16,
            learning_rate: 0.05,
            weight_limit: 2.0,
            temperature: 1.0,
        }
    }
}

/// Two-odor, two-action agent with contiguous action-major synaptic weights.
pub struct Agent {
    config: Config,
    weights: Vec<f32>,
    active: Vec<usize>,
    eligibility: Vec<f32>,
    rng: u64,
    pending: bool,
}

impl Agent {
    pub fn new(seed: u64) -> Self {
        Self::with_config(seed, Config::default())
    }

    pub fn with_config(seed: u64, config: Config) -> Self {
        assert!(
            config.kcs >= 2 && config.kcs.is_multiple_of(2),
            "kcs must be positive and even"
        );
        assert!(
            config.active_kcs > 0 && config.active_kcs <= config.kcs / 2,
            "active_kcs must fit in one disjoint cue pool"
        );
        assert!(config.learning_rate.is_finite() && config.learning_rate > 0.0);
        assert!(config.weight_limit.is_finite() && config.weight_limit > 0.0);
        assert!(config.temperature.is_finite() && config.temperature > 0.0);
        Self {
            config,
            weights: vec![0.0; ACTIONS * config.kcs],
            active: vec![0; config.active_kcs],
            eligibility: vec![0.0; ACTIONS * config.active_kcs],
            rng: if seed == 0 {
                0x9e37_79b9_7f4a_7c15
            } else {
                seed
            },
            pending: false,
        }
    }

    pub fn config(&self) -> Config {
        self.config
    }

    /// Select an action. Training samples from the policy and records its
    /// eligibility; evaluation returns the greedy action and never learns.
    pub fn action(&mut self, cue: usize, learn: bool) -> usize {
        assert!(cue < CUES, "cue must be 0 or 1");
        self.pending = false;
        self.encode(cue);
        let mut logits = [0.0f32; ACTIONS];
        for (a, logit) in logits.iter_mut().enumerate() {
            let row = &self.weights[a * self.config.kcs..(a + 1) * self.config.kcs];
            for &kc in &self.active {
                *logit += row[kc];
            }
        }
        if !learn {
            return usize::from(logits[1] > logits[0]);
        }

        let d = ((logits[1] - logits[0]) / self.config.temperature).clamp(-80.0, 80.0);
        let p1 = 1.0 / (1.0 + (-d).exp());
        let probabilities = [1.0 - p1, p1];
        let chosen = usize::from(self.uniform() < p1);
        for (a, &probability) in probabilities.iter().enumerate() {
            let e = f32::from((a == chosen) as u8) - probability;
            self.eligibility[a * self.config.active_kcs..(a + 1) * self.config.active_kcs].fill(e);
        }
        self.pending = true;
        chosen
    }

    /// Apply dopamine-like reward prediction error to the preceding training
    /// action. A zero baseline keeps the rule valid for signed or 0/1 rewards.
    pub fn reward(&mut self, reward: f32) {
        assert!(reward.is_finite(), "reward must be finite");
        assert!(self.pending, "reward requires a preceding training action");
        let scale = self.config.learning_rate * reward / self.config.temperature;
        for a in 0..ACTIONS {
            let row = &mut self.weights[a * self.config.kcs..(a + 1) * self.config.kcs];
            let eligibility =
                &self.eligibility[a * self.config.active_kcs..(a + 1) * self.config.active_kcs];
            for (&kc, &e) in self.active.iter().zip(eligibility) {
                row[kc] = (row[kc] + scale * e)
                    .clamp(-self.config.weight_limit, self.config.weight_limit);
            }
        }
        self.pending = false;
    }

    /// Clear transient eligibility without changing learned synapses.
    pub fn reset_state(&mut self) {
        self.pending = false;
        self.eligibility.fill(0.0);
    }

    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Deterministic sparse, disjoint cue patterns. The stride is coprime to
    /// the pool size, so each active cell is unique for any valid configuration.
    fn encode(&mut self, cue: usize) {
        let pool = self.config.kcs / 2;
        let offset = cue * pool;
        for (i, kc) in self.active.iter_mut().enumerate() {
            *kc = offset + (i.wrapping_mul(pool - 1).wrapping_add(19) % pool);
        }
    }

    fn uniform(&mut self) -> f32 {
        // xorshift64*: deterministic, fast, and adequate for this demo.
        self.rng ^= self.rng >> 12;
        self.rng ^= self.rng << 25;
        self.rng ^= self.rng >> 27;
        let bits = self.rng.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 40;
        (bits as f32) * (1.0 / 16_777_216.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn learns_opposing_odors_across_seeds() {
        for seed in 1..=12 {
            let mut agent = Agent::new(seed);
            let mut before = 0;
            for cue in 0..CUES {
                before += usize::from(agent.action(cue, false) == cue);
            }
            for episode in 0..300 {
                let cue = episode % CUES;
                let action = agent.action(cue, true);
                agent.reward(if action == cue { 1.0 } else { -1.0 });
            }
            let after = (0..CUES)
                .filter(|&cue| agent.action(cue, false) == cue)
                .count();
            assert!(
                after > before,
                "seed {seed}: before={before}, after={after}"
            );
            assert_eq!(after, CUES);
        }
    }

    #[test]
    fn evaluation_is_read_only_and_deterministic() {
        let mut a = Agent::new(7);
        let mut b = Agent::new(7);
        for i in 0..50 {
            let cue = i % CUES;
            assert_eq!(a.action(cue, true), b.action(cue, true));
            a.reward(1.0);
            b.reward(1.0);
        }
        let original = a.weights().to_vec();
        for _ in 0..100 {
            a.action(0, false);
            a.action(1, false);
        }
        assert_eq!(a.weights(), original);
        assert_eq!(a.weights(), b.weights());
    }

    #[test]
    fn weights_are_bounded() {
        let mut a = Agent::new(42);
        for _ in 0..500 {
            a.action(0, true);
            a.reward(100.0);
        }
        assert!(a
            .weights()
            .iter()
            .all(|w| w.abs() <= a.config().weight_limit));
    }

    #[test]
    #[should_panic(expected = "cue must be 0 or 1")]
    fn invalid_cue_panics() {
        Agent::new(1).action(2, false);
    }

    #[test]
    #[should_panic(expected = "reward requires")]
    fn reward_without_action_panics() {
        Agent::new(1).reward(1.0);
    }
}
