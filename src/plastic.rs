//! Topology-constrained rate-neuron policy with reward-modulated plastic edges.
//! This is a rate model, not a spiking or biophysically exact mushroom body.
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Clone, Copy, Debug)]
pub struct PlasticConfig {
    pub actions: usize,
    pub learning_rate: f32,
    pub logit_gain: f32,
    pub active_fraction: f32,
    pub weight_limit: f32,
    pub homeostasis: bool,
}
impl Default for PlasticConfig {
    fn default() -> Self {
        Self {
            actions: 2,
            learning_rate: 0.02,
            logit_gain: 6.0,
            active_fraction: 0.1,
            weight_limit: 1.0,
            homeostasis: true,
        }
    }
}
impl PlasticConfig {
    fn validate(self) -> Result<(), String> {
        if self.actions < 2 {
            return Err("actions must be >= 2".into());
        }
        for (name, value) in [
            ("learning_rate", self.learning_rate),
            ("logit_gain", self.logit_gain),
            ("active_fraction", self.active_fraction),
            ("weight_limit", self.weight_limit),
        ] {
            if !value.is_finite() {
                return Err(format!("{name} must be finite"));
            }
        }
        if self.learning_rate < 0.0
            || self.logit_gain <= 0.0
            || !(0.0 < self.active_fraction && self.active_fraction <= 1.0)
            || self.weight_limit <= 0.0
        {
            return Err("invalid config range".into());
        }
        Ok(())
    }
}
#[derive(Clone, Copy, Debug)]
struct RawEdge {
    pre: u64,
    post: u64,
    count: u64,
    sign: i8,
}
#[derive(Clone, Copy, Debug)]
struct Edge {
    pre: usize,
    post: usize,
    count: u64,
    sign: i8,
    weight: f32,
}

pub struct Engine {
    config: PlasticConfig,
    input_ids: Vec<u64>,
    hidden_ids: Vec<u64>,
    output_ids: Vec<u64>,
    input_edges: Vec<Edge>,
    plastic_edges: Vec<Edge>,
    output_actions: Vec<usize>,
    output_gains: Vec<f32>,
    hidden: Vec<f32>,
    output: Vec<f32>,
    probabilities: Vec<f32>,
    group_counts: Vec<usize>,
    reward_coeff: Vec<f32>,
    output_l1: Vec<f32>,
    ranks: Vec<usize>,
    pending: bool,
    batch_gradient: Vec<f32>,
    batch_count: usize,
}
impl Engine {
    pub fn load<P: AsRef<Path>, Q: AsRef<Path>>(
        input_edges_path: P,
        plastic_edges_path: Q,
        config: PlasticConfig,
    ) -> Result<Self, String> {
        config.validate()?;
        let input_raw = read_edges(input_edges_path.as_ref())?;
        let plastic_raw = read_edges(plastic_edges_path.as_ref())?;
        Self::from_raw(input_raw, plastic_raw, config)
    }
    fn from_raw(
        input_raw: Vec<RawEdge>,
        plastic_raw: Vec<RawEdge>,
        config: PlasticConfig,
    ) -> Result<Self, String> {
        if input_raw.is_empty() || plastic_raw.is_empty() {
            return Err("both edge layers must be nonempty".into());
        }
        let input_ids: Vec<_> = input_raw
            .iter()
            .map(|e| e.pre)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        // Plastic presynaptic cells without a fixed input edge remain present and silent.
        let hidden_ids: Vec<_> = input_raw
            .iter()
            .map(|e| e.post)
            .chain(plastic_raw.iter().map(|e| e.pre))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let output_ids: Vec<_> = plastic_raw
            .iter()
            .map(|e| e.post)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        if config.actions > output_ids.len() {
            return Err("actions cannot exceed output cells".into());
        }
        let im = index(&input_ids);
        let hm = index(&hidden_ids);
        let om = index(&output_ids);
        let mut input_edges = Vec::with_capacity(input_raw.len());
        let mut plastic_edges = Vec::with_capacity(plastic_raw.len());
        let mut input_totals = vec![0f64; hidden_ids.len()];
        let mut output_totals = vec![0f64; output_ids.len()];
        for e in &input_raw {
            input_totals[hm[&e.post]] += e.count as f64;
        }
        for e in &plastic_raw {
            output_totals[om[&e.post]] += e.count as f64;
        }
        for e in input_raw {
            let post = hm[&e.post];
            input_edges.push(Edge {
                pre: im[&e.pre],
                post,
                count: e.count,
                sign: e.sign,
                weight: e.sign as f32 * (e.count as f64 / input_totals[post]) as f32,
            });
        }
        for e in plastic_raw {
            let pre = hm[&e.pre];
            let post = om[&e.post];
            plastic_edges.push(Edge {
                pre,
                post,
                count: e.count,
                sign: e.sign,
                weight: e.sign as f32
                    * ((e.count as f64 / output_totals[post]) as f32).min(config.weight_limit),
            });
        }
        let output_actions: Vec<_> = (0..output_ids.len()).map(|i| i % config.actions).collect();
        let plastic_edge_count = plastic_edges.len();
        let mut engine = Self {
            config,
            input_ids,
            hidden_ids,
            output_ids,
            input_edges,
            plastic_edges,
            output_actions,
            output_gains: vec![1.0; om.len()],
            hidden: vec![0.0; hm.len()],
            output: vec![0.0; om.len()],
            probabilities: vec![0.0; config.actions],
            group_counts: vec![0; config.actions],
            reward_coeff: vec![0.0; config.actions],
            output_l1: vec![0.0; om.len()],
            ranks: (0..hm.len()).collect(),
            pending: false,
            batch_gradient: vec![0.0; plastic_edge_count],
            batch_count: 0,
        };
        engine.recount_groups();
        Ok(engine)
    }
    pub fn config(&self) -> PlasticConfig {
        self.config
    }
    /// True while a forward trial or accumulated gradient batch is unapplied.
    pub fn has_pending_work(&self) -> bool {
        self.pending || self.batch_count > 0
    }
    pub fn input_ids(&self) -> &[u64] {
        &self.input_ids
    }
    pub fn hidden_ids(&self) -> &[u64] {
        &self.hidden_ids
    }
    pub fn output_ids(&self) -> &[u64] {
        &self.output_ids
    }
    pub fn input_edge_count(&self) -> usize {
        self.input_edges.len()
    }
    pub fn plastic_edge_count(&self) -> usize {
        self.plastic_edges.len()
    }
    pub fn output_actions(&self) -> &[usize] {
        &self.output_actions
    }
    /// Fixed signed gains in the external action readout; plastic edge signs are unchanged.
    pub fn output_gains(&self) -> &[f32] {
        &self.output_gains
    }
    pub fn set_output_gains(&mut self, gains: &[f32]) -> Result<(), String> {
        if self.batch_count > 0 {
            return Err("apply accumulated batch before changing output gains".into());
        }
        if gains.len() != self.output_ids.len()
            || gains
                .iter()
                .any(|&x| !x.is_finite() || !(-1.0..=1.0).contains(&x))
        {
            return Err("output gains must match outputs and be finite in [-1, 1]".into());
        }
        let mut mass = vec![0.0f32; self.config.actions];
        for (o, &g) in self.output_actions.iter().enumerate() {
            mass[g] += gains[o].abs();
        }
        if mass.contains(&0.0) {
            return Err("every action must have nonzero output gain mass".into());
        }
        self.output_gains.copy_from_slice(gains);
        self.pending = false;
        Ok(())
    }
    pub fn hidden_activity(&self) -> &[f32] {
        &self.hidden
    }
    pub fn weights(&self) -> impl Iterator<Item = f32> + '_ {
        self.plastic_edges.iter().map(|e| e.weight)
    }
    pub fn set_output_actions(&mut self, groups: &[usize]) -> Result<(), String> {
        if self.batch_count > 0 {
            return Err("apply accumulated batch before changing output actions".into());
        }
        if groups.len() != self.output_ids.len() || groups.iter().any(|&g| g >= self.config.actions)
        {
            return Err("output action mapping has wrong length or invalid action".into());
        }
        let mut mass = vec![0.0f32; self.config.actions];
        for (o, &g) in groups.iter().enumerate() {
            mass[g] += self.output_gains[o].abs();
        }
        if mass.contains(&0.0) {
            return Err("every action must have nonzero output gain mass".into());
        }
        self.output_actions.copy_from_slice(groups);
        self.recount_groups();
        self.pending = false;
        Ok(())
    }
    fn recount_groups(&mut self) {
        self.group_counts.fill(0);
        for &g in &self.output_actions {
            self.group_counts[g] += 1;
        }
    }
    pub fn forward(&mut self, input: &[f32]) -> &[f32] {
        assert_eq!(
            input.len(),
            self.input_ids.len(),
            "input length must match input_ids"
        );
        assert!(input.iter().all(|x| x.is_finite()), "input must be finite");
        let max_abs = input.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
        self.hidden.fill(0.0);
        if max_abs > 16.0 {
            for e in &self.input_edges {
                self.hidden[e.post] += e.weight * (input[e.pre] / max_abs);
            }
        } else {
            for e in &self.input_edges {
                self.hidden[e.post] += e.weight * input[e.pre];
            }
        }
        assert!(
            self.hidden.iter().all(|v| v.is_finite()),
            "hidden activity overflow"
        );
        for h in &mut self.hidden {
            *h = h.max(0.0);
        }
        let n = self.hidden.len();
        let keep = ((n as f32 * self.config.active_fraction).ceil() as usize).clamp(1, n);
        if keep < n {
            for (i, rank) in self.ranks.iter_mut().enumerate() {
                *rank = i;
            }
            let h = &self.hidden;
            self.ranks.select_nth_unstable_by(keep - 1, |&a, &b| {
                h[b].total_cmp(&h[a]).then_with(|| a.cmp(&b))
            });
            // Exactly `keep` winners, including deterministic tie handling.
            for &i in &self.ranks[keep..] {
                self.hidden[i] = 0.0;
            }
        }
        let max = self.hidden.iter().copied().fold(0.0f32, f32::max);
        if max > 0.0 {
            for h in &mut self.hidden {
                *h /= max;
            }
        }
        self.output.fill(0.0);
        for e in &self.plastic_edges {
            self.output[e.post] += e.weight * self.hidden[e.pre];
        }
        assert!(
            self.output.iter().all(|v| v.is_finite()),
            "output activity overflow"
        );
        self.probabilities.fill(0.0);
        for (o, &g) in self.output_actions.iter().enumerate() {
            self.probabilities[g] += self.output_gains[o] * self.output[o];
        }
        for (a, p) in self.probabilities.iter_mut().enumerate() {
            if self.group_counts[a] > 0 {
                *p *= self.config.logit_gain / self.group_counts[a] as f32;
            }
        }
        assert!(
            self.probabilities.iter().all(|v| v.is_finite()),
            "action logit overflow"
        );
        let peak = self
            .probabilities
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for p in &mut self.probabilities {
            *p = (*p - peak).exp();
            sum += *p;
        }
        for p in &mut self.probabilities {
            *p /= sum;
        }
        assert!(
            self.probabilities.iter().all(|v| v.is_finite()),
            "action probability overflow"
        );
        self.pending = true;
        &self.probabilities
    }
    pub fn reward(&mut self, action: usize, reward: f32) -> Result<(), String> {
        if self.batch_count > 0 {
            return Err("reward cannot mix online updates with an accumulated batch".into());
        }
        if !self.pending {
            return Err("reward requires one unmatched forward call".into());
        }
        if action >= self.config.actions || !reward.is_finite() {
            return Err("invalid action or reward".into());
        }
        if reward == 0.0 {
            self.pending = false;
            return Ok(());
        }
        for g in 0..self.config.actions {
            let indicator = if g == action { 1.0 } else { 0.0 };
            self.reward_coeff[g] = self.config.learning_rate
                * reward
                * (indicator - self.probabilities[g])
                * self.config.logit_gain
                / self.group_counts[g] as f32;
        }
        for e in &mut self.plastic_edges {
            if e.sign == 0 || self.hidden[e.pre] == 0.0 {
                continue;
            }
            let g = self.output_actions[e.post];
            let delta = self.reward_coeff[g] * self.output_gains[e.post] * self.hidden[e.pre];
            let next = e.weight + delta;
            e.weight = if e.sign > 0 {
                next.clamp(0.0, self.config.weight_limit)
            } else {
                next.clamp(-self.config.weight_limit, 0.0)
            };
        }
        if self.config.homeostasis {
            self.output_l1.fill(0.0);
            for e in &self.plastic_edges {
                self.output_l1[e.post] += e.weight.abs();
            }
            for e in &mut self.plastic_edges {
                let total = self.output_l1[e.post];
                if total > 1.0 {
                    e.weight /= total;
                }
            }
        }
        self.pending = false;
        Ok(())
    }
    /// Accumulate a teacher/reward gradient at the current weights without updating them.
    /// Every call consumes exactly one preceding forward trial, including zero reward.
    pub fn accumulate_reward(&mut self, action: usize, reward: f32) -> Result<(), String> {
        if !self.pending {
            return Err("accumulate_reward requires one unmatched forward call".into());
        }
        if action >= self.config.actions || !reward.is_finite() {
            return Err("invalid action or reward".into());
        }
        if self.batch_count == usize::MAX {
            return Err("batch count overflow".into());
        }
        if reward != 0.0 {
            for g in 0..self.config.actions {
                let indicator = if g == action { 1.0 } else { 0.0 };
                self.reward_coeff[g] = self.config.learning_rate
                    * reward
                    * (indicator - self.probabilities[g])
                    * self.config.logit_gain
                    / self.group_counts[g] as f32;
            }
            for (gradient, e) in self.batch_gradient.iter_mut().zip(&self.plastic_edges) {
                if e.sign != 0 && self.hidden[e.pre] != 0.0 {
                    let g = self.output_actions[e.post];
                    *gradient +=
                        self.reward_coeff[g] * self.output_gains[e.post] * self.hidden[e.pre];
                }
            }
        }
        self.batch_count += 1;
        self.pending = false;
        Ok(())
    }
    /// Apply the mean gradient of the accumulated trials, then project weights once.
    pub fn apply_batch(&mut self) -> Result<(), String> {
        if self.pending {
            return Err("apply_batch requires a completed trial".into());
        }
        if self.batch_count == 0 {
            return Err("apply_batch requires accumulated trials".into());
        }
        let divisor = self.batch_count as f32;
        if !divisor.is_finite() || self.batch_gradient.iter().any(|x| !x.is_finite()) {
            return Err("batch gradient is not finite".into());
        }
        for (e, &gradient) in self.plastic_edges.iter_mut().zip(&self.batch_gradient) {
            if e.sign == 0 {
                continue;
            }
            let next = e.weight + gradient / divisor;
            e.weight = if e.sign > 0 {
                next.clamp(0.0, self.config.weight_limit)
            } else {
                next.clamp(-self.config.weight_limit, 0.0)
            };
        }
        if self.config.homeostasis {
            self.output_l1.fill(0.0);
            for e in &self.plastic_edges {
                self.output_l1[e.post] += e.weight.abs();
            }
            for e in &mut self.plastic_edges {
                let total = self.output_l1[e.post];
                if total > 1.0 {
                    e.weight /= total;
                }
            }
        }
        self.batch_gradient.fill(0.0);
        self.batch_count = 0;
        Ok(())
    }
    /// Save a self-contained, versioned safetensors checkpoint at a completed trial boundary.
    pub fn save_checkpoint<P: AsRef<Path>>(&self, path: P) -> Result<(), String> {
        if self.pending || self.batch_count > 0 {
            return Err("checkpoint requires a completed trial and applied batch".into());
        }
        use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
        let data: Vec<(&str, Dtype, Vec<u8>)> = vec![
            (
                "input_ids",
                Dtype::U64,
                encode_u64(self.input_ids.iter().copied()),
            ),
            (
                "hidden_ids",
                Dtype::U64,
                encode_u64(self.hidden_ids.iter().copied()),
            ),
            (
                "output_ids",
                Dtype::U64,
                encode_u64(self.output_ids.iter().copied()),
            ),
            (
                "input_pre",
                Dtype::U64,
                encode_u64(self.input_edges.iter().map(|e| e.pre as u64)),
            ),
            (
                "input_post",
                Dtype::U64,
                encode_u64(self.input_edges.iter().map(|e| e.post as u64)),
            ),
            (
                "input_count",
                Dtype::U64,
                encode_u64(self.input_edges.iter().map(|e| e.count)),
            ),
            (
                "input_sign",
                Dtype::I8,
                self.input_edges.iter().map(|e| e.sign as u8).collect(),
            ),
            (
                "plastic_pre",
                Dtype::U64,
                encode_u64(self.plastic_edges.iter().map(|e| e.pre as u64)),
            ),
            (
                "plastic_post",
                Dtype::U64,
                encode_u64(self.plastic_edges.iter().map(|e| e.post as u64)),
            ),
            (
                "plastic_count",
                Dtype::U64,
                encode_u64(self.plastic_edges.iter().map(|e| e.count)),
            ),
            (
                "plastic_sign",
                Dtype::I8,
                self.plastic_edges.iter().map(|e| e.sign as u8).collect(),
            ),
            (
                "plastic_weight",
                Dtype::F32,
                encode_f32(self.plastic_edges.iter().map(|e| e.weight)),
            ),
            (
                "output_actions",
                Dtype::U64,
                encode_u64(self.output_actions.iter().map(|&g| g as u64)),
            ),
            (
                "output_gains",
                Dtype::F32,
                encode_f32(self.output_gains.iter().copied()),
            ),
        ];
        let views: Vec<_> = data
            .iter()
            .map(|(name, dtype, bytes)| {
                TensorView::new(*dtype, vec![bytes.len() / (dtype.bitsize() / 8)], bytes)
                    .map(|v| (*name, v))
                    .map_err(|e| e.to_string())
            })
            .collect::<Result<_, _>>()?;
        let c = self.config;
        let metadata = std::collections::HashMap::from([
            ("format".into(), "nobi.plastic".into()),
            ("version".into(), "4".into()),
            ("actions".into(), c.actions.to_string()),
            ("learning_rate".into(), c.learning_rate.to_string()),
            ("logit_gain".into(), c.logit_gain.to_string()),
            ("active_fraction".into(), c.active_fraction.to_string()),
            ("weight_limit".into(), c.weight_limit.to_string()),
            ("homeostasis".into(), c.homeostasis.to_string()),
        ]);
        let target = path.as_ref();
        let mut temporary = target.as_os_str().to_os_string();
        temporary.push(format!(
            ".{}.{}.tmp",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_err(|e| e.to_string())?
                .as_nanos()
        ));
        let temporary = std::path::PathBuf::from(temporary);
        let result = serialize_to_file(views, Some(metadata), &temporary)
            .map_err(|e| e.to_string())
            .and_then(|_| std::fs::rename(&temporary, target).map_err(|e| e.to_string()));
        if result.is_err() {
            let _ = std::fs::remove_file(&temporary);
        }
        result
    }
    /// Discard only the latest forward trial; accumulated batch gradients remain intact.
    pub fn clear_pending(&mut self) {
        self.pending = false;
    }
    /// Intentionally discard all unapplied batch gradients and any current forward trial.
    pub fn discard_batch(&mut self) {
        self.batch_gradient.fill(0.0);
        self.batch_count = 0;
        self.pending = false;
    }
    pub fn load_checkpoint<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        use safetensors::SafeTensors;
        let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
        let (_, header) = SafeTensors::read_metadata(&bytes).map_err(|e| e.to_string())?;
        let metadata = header
            .metadata()
            .as_ref()
            .ok_or("missing checkpoint metadata")?;
        let field = |key: &str| {
            metadata
                .get(key)
                .map(String::as_str)
                .ok_or_else(|| format!("missing checkpoint metadata: {key}"))
        };
        let version = field("version")?;
        if field("format")? != "nobi.plastic" || (version != "3" && version != "4") {
            return Err("unsupported checkpoint format or version".into());
        }
        let config = PlasticConfig {
            actions: parse(field("actions")?)?,
            learning_rate: parse(field("learning_rate")?)?,
            logit_gain: parse(field("logit_gain")?)?,
            active_fraction: parse(field("active_fraction")?)?,
            weight_limit: parse(field("weight_limit")?)?,
            homeostasis: parse(field("homeostasis")?)?,
        };
        config.validate()?;
        let st = SafeTensors::deserialize(&bytes).map_err(|e| e.to_string())?;
        if st.len() != if version == "4" { 14 } else { 13 } {
            return Err("unexpected checkpoint tensor set".into());
        }
        let ii = read_u64(&st, "input_ids")?;
        let hi = read_u64(&st, "hidden_ids")?;
        let oi = read_u64(&st, "output_ids")?;
        let ip = read_u64(&st, "input_pre")?;
        let iq = read_u64(&st, "input_post")?;
        let ic = read_u64(&st, "input_count")?;
        let is = read_i8(&st, "input_sign")?;
        let pp = read_u64(&st, "plastic_pre")?;
        let pq = read_u64(&st, "plastic_post")?;
        let pc = read_u64(&st, "plastic_count")?;
        let ps = read_i8(&st, "plastic_sign")?;
        let pw = read_f32(&st, "plastic_weight")?;
        let groups = read_u64(&st, "output_actions")?;
        let gains = if version == "4" {
            Some(read_f32(&st, "output_gains")?)
        } else {
            None
        };
        if ip.len() != iq.len()
            || ip.len() != ic.len()
            || ip.len() != is.len()
            || pp.len() != pq.len()
            || pp.len() != pc.len()
            || pp.len() != ps.len()
            || pp.len() != pw.len()
            || oi.len() != groups.len()
        {
            return Err("checkpoint tensor lengths disagree".into());
        }
        let mut ir = Vec::with_capacity(ip.len());
        let mut pr = Vec::with_capacity(pp.len());
        for j in 0..ip.len() {
            let a = *ii
                .get(usize::try_from(ip[j]).map_err(|_| "input index overflow")?)
                .ok_or("input index out of range")?;
            let b = *hi
                .get(usize::try_from(iq[j]).map_err(|_| "hidden index overflow")?)
                .ok_or("hidden index out of range")?;
            ir.push(RawEdge {
                pre: a,
                post: b,
                count: ic[j],
                sign: is[j],
            });
            validate_raw(ir[j])?;
        }
        for j in 0..pp.len() {
            let a = *hi
                .get(usize::try_from(pp[j]).map_err(|_| "hidden index overflow")?)
                .ok_or("hidden index out of range")?;
            let b = *oi
                .get(usize::try_from(pq[j]).map_err(|_| "output index overflow")?)
                .ok_or("output index out of range")?;
            pr.push(RawEdge {
                pre: a,
                post: b,
                count: pc[j],
                sign: ps[j],
            });
            validate_raw(pr[j])?;
        }
        let mut engine = Self::from_raw(ir, pr, config)?;
        if engine.input_ids != ii || engine.hidden_ids != hi || engine.output_ids != oi {
            return Err("checkpoint ID tensors disagree with topology".into());
        }
        for (e, w) in engine.plastic_edges.iter_mut().zip(pw) {
            if !w.is_finite()
                || (e.sign > 0 && !(0.0..=config.weight_limit).contains(&w))
                || (e.sign < 0 && !(-config.weight_limit..=0.0).contains(&w))
                || (e.sign == 0 && w != 0.0)
            {
                return Err("checkpoint weight violates sign or bounds".into());
            }
            e.weight = w;
        }
        if config.homeostasis {
            let mut totals = vec![0.0f64; engine.output_ids.len()];
            for e in &engine.plastic_edges {
                totals[e.post] += f64::from(e.weight.abs());
            }
            if totals.iter().any(|&total| total > 1.0001) {
                return Err("checkpoint output L1 mass exceeds homeostasis budget".into());
            }
        }
        let groups: Vec<usize> = groups
            .into_iter()
            .map(|g| usize::try_from(g).map_err(|_| "action index overflow".to_string()))
            .collect::<Result<_, _>>()?;
        engine.set_output_actions(&groups)?;
        if let Some(gains) = gains {
            engine.set_output_gains(&gains)?;
        }
        Ok(engine)
    }
}
fn index(ids: &[u64]) -> BTreeMap<u64, usize> {
    ids.iter().enumerate().map(|(i, &id)| (id, i)).collect()
}
fn parse<T: std::str::FromStr>(s: &str) -> Result<T, String> {
    s.parse().map_err(|_| format!("invalid field: {s}"))
}
fn raw(a: &str, b: &str, c: &str, d: &str) -> Result<RawEdge, String> {
    let e = RawEdge {
        pre: parse(a)?,
        post: parse(b)?,
        count: parse(c)?,
        sign: parse(d)?,
    };
    validate_raw(e)?;
    Ok(e)
}
fn validate_raw(e: RawEdge) -> Result<(), String> {
    if e.pre == 0 || e.post == 0 || e.count == 0 || !(-1..=1).contains(&e.sign) {
        return Err("edge IDs and count must be positive and sign must be -1, 0, or 1".into());
    }
    Ok(())
}
fn encode_u64(values: impl Iterator<Item = u64>) -> Vec<u8> {
    values.flat_map(u64::to_le_bytes).collect()
}
fn encode_f32(values: impl Iterator<Item = f32>) -> Vec<u8> {
    values.flat_map(f32::to_le_bytes).collect()
}
fn read_tensor<'a>(
    st: &'a safetensors::SafeTensors<'a>,
    name: &str,
    dtype: safetensors::Dtype,
) -> Result<safetensors::tensor::TensorView<'a>, String> {
    let t = st.tensor(name).map_err(|e| e.to_string())?;
    if t.dtype() != dtype || t.shape().len() != 1 {
        return Err(format!("bad tensor dtype or shape: {name}"));
    }
    Ok(t)
}
fn read_u64(st: &safetensors::SafeTensors<'_>, name: &str) -> Result<Vec<u64>, String> {
    Ok(read_tensor(st, name, safetensors::Dtype::U64)?
        .data()
        .chunks_exact(8)
        .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
        .collect())
}
fn read_i8(st: &safetensors::SafeTensors<'_>, name: &str) -> Result<Vec<i8>, String> {
    Ok(read_tensor(st, name, safetensors::Dtype::I8)?
        .data()
        .iter()
        .map(|&b| b as i8)
        .collect())
}
fn read_f32(st: &safetensors::SafeTensors<'_>, name: &str) -> Result<Vec<f32>, String> {
    Ok(read_tensor(st, name, safetensors::Dtype::F32)?
        .data()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}
fn read_edges(path: &Path) -> Result<Vec<RawEdge>, String> {
    let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut result = Vec::new();
    for (n, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|e| e.to_string())?;
        if line.trim().is_empty() {
            continue;
        }
        let x: Vec<_> = line.split_whitespace().collect();
        if x.len() != 4 {
            return Err(format!(
                "{}:{}: expected pre post count sign",
                path.display(),
                n + 1
            ));
        }
        result.push(
            raw(x[0], x[1], x[2], x[3])
                .map_err(|e| format!("{}:{}: {e}", path.display(), n + 1))?,
        );
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    fn path(label: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "novi_plastic_{}_{}_{}",
            label,
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ))
    }
    fn fixture() -> Engine {
        let a = path("i");
        let b = path("p");
        fs::write(&a, "1\t10\t3\t1\n2\t20\t2\t1\n").unwrap();
        fs::write(&b, "10\t100\t1\t1\n10\t200\t1\t-1\n20\t100\t1\t0\n").unwrap();
        let e = Engine::load(
            &a,
            &b,
            PlasticConfig {
                active_fraction: 1.0,
                homeostasis: false,
                ..Default::default()
            },
        )
        .unwrap();
        fs::remove_file(a).unwrap();
        fs::remove_file(b).unwrap();
        e
    }
    #[test]
    fn gradient_matches_finite_difference_and_reward_is_single_use() {
        let mut e = fixture();
        e.plastic_edges[0].weight = 0.5;
        let before: Vec<_> = e.weights().collect();
        let p = e.forward(&[1.0, 0.0]).to_vec();
        assert_eq!(before, e.weights().collect::<Vec<_>>());
        let eps = 1e-3;
        e.plastic_edges[0].weight = before[0] + eps;
        let plus = e.forward(&[1.0, 0.0])[0].ln();
        e.plastic_edges[0].weight = before[0] - eps;
        let minus = e.forward(&[1.0, 0.0])[0].ln();
        let numeric = (plus - minus) / (2.0 * eps);
        e.plastic_edges[0].weight = before[0];
        e.forward(&[1.0, 0.0]);
        e.reward(0, 1.0).unwrap();
        let after: Vec<_> = e.weights().collect();
        let analytic = (after[0] - before[0]) / e.config.learning_rate;
        assert!((numeric - analytic).abs() < 0.001, "{numeric} {analytic}");
        assert_eq!(after[1], before[1]); // inhibitory edge cannot cross zero
        assert_eq!(after[2], 0.0); // unknown sign is frozen
        assert!(e.reward(0, 1.0).is_err());
        assert!(p[0] > p[1]);
    }
    #[test]
    fn batch_applies_mean_of_finite_difference_gradients() {
        let mut e = fixture();
        e.plastic_edges[0].weight = 0.5;
        let original = e.plastic_edges[0].weight;
        let eps = 1e-3;
        e.plastic_edges[0].weight = original + eps;
        let plus = (e.forward(&[1.0, 0.0])[0].ln() + e.forward(&[0.0, 1.0])[0].ln()) / 2.0;
        e.plastic_edges[0].weight = original - eps;
        let minus = (e.forward(&[1.0, 0.0])[0].ln() + e.forward(&[0.0, 1.0])[0].ln()) / 2.0;
        let numeric = (plus - minus) / (2.0 * eps);
        e.plastic_edges[0].weight = original;
        e.forward(&[1.0, 0.0]);
        e.accumulate_reward(0, 1.0).unwrap();
        assert!(e.reward(0, 1.0).is_err());
        assert!(e.save_checkpoint(path("unapplied_batch")).is_err());
        e.forward(&[0.0, 1.0]);
        e.accumulate_reward(0, 1.0).unwrap();
        e.apply_batch().unwrap();
        let analytic = (e.plastic_edges[0].weight - original) / e.config.learning_rate;
        assert!((numeric - analytic).abs() < 0.001, "{numeric} {analytic}");
        assert_eq!(e.batch_count, 0);
        assert!(e.batch_gradient.iter().all(|&x| x == 0.0));
    }
    #[test]
    fn one_accumulated_sample_matches_online_update() {
        let mut online = fixture();
        let mut batch = fixture();
        online.plastic_edges[0].weight = 0.5;
        batch.plastic_edges[0].weight = 0.5;
        online.forward(&[1.0, 0.0]);
        online.reward(0, 1.0).unwrap();
        batch.forward(&[1.0, 0.0]);
        batch.accumulate_reward(0, 1.0).unwrap();
        batch.apply_batch().unwrap();
        assert_eq!(
            online.weights().collect::<Vec<_>>(),
            batch.weights().collect::<Vec<_>>()
        );
    }
    #[test]
    fn decoder_changes_are_atomic_during_batch_and_discard_recovers() {
        let mut e = fixture();
        let groups = e.output_actions().to_vec();
        let gains = e.output_gains().to_vec();
        let weights = e.weights().collect::<Vec<_>>();
        e.forward(&[1.0, 0.0]);
        e.accumulate_reward(0, 1.0).unwrap();
        let swapped: Vec<_> = groups.iter().map(|&g| 1 - g).collect();
        assert!(e.set_output_actions(&swapped).is_err());
        assert!(e.set_output_gains(&[-1.0, 1.0]).is_err());
        assert_eq!(e.output_actions(), groups);
        assert_eq!(e.output_gains(), gains);
        assert_eq!(e.weights().collect::<Vec<_>>(), weights);
        e.clear_pending();
        assert_eq!(e.batch_count, 1);
        assert!(e.save_checkpoint(path("still_unapplied")).is_err());
        e.discard_batch();
        assert_eq!(e.batch_count, 0);
        assert!(e.batch_gradient.iter().all(|&g| g == 0.0));
        assert!(e.apply_batch().is_err());
        let saved = path("discarded_batch");
        e.save_checkpoint(&saved).unwrap();
        fs::remove_file(saved).unwrap();
        e.set_output_actions(&swapped).unwrap();
    }
    #[test]
    fn checkpoint_roundtrip_and_resume() {
        let mut e = fixture();
        e.forward(&[1.0, 0.0]);
        e.reward(1, 0.5).unwrap();
        let p = path("checkpoint");
        e.save_checkpoint(&p).unwrap();
        let bytes = fs::read(&p).unwrap();
        let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
        assert_eq!(st.len(), 14);
        assert_eq!(
            st.tensor("plastic_weight").unwrap().dtype(),
            safetensors::Dtype::F32
        );
        let mut restored = Engine::load_checkpoint(&p).unwrap();
        let truncated = path("truncated");
        fs::write(&truncated, &bytes[..bytes.len() - 1]).unwrap();
        assert!(Engine::load_checkpoint(&truncated).is_err());
        fs::remove_file(truncated).unwrap();
        fs::remove_file(p).unwrap();
        assert_eq!(
            e.weights().collect::<Vec<_>>(),
            restored.weights().collect::<Vec<_>>()
        );
        assert_eq!(e.forward(&[0.4, 0.7]), restored.forward(&[0.4, 0.7]));
        e.reward(0, -1.0).unwrap();
        restored.reward(0, -1.0).unwrap();
        assert_eq!(
            e.weights().collect::<Vec<_>>(),
            restored.weights().collect::<Vec<_>>()
        );
    }
    #[test]
    fn homeostasis_bounds_output_mass_and_probabilities() {
        let mut e = fixture();
        e.config.homeostasis = true;
        for t in 0..1000 {
            let p = e.forward(&[1.0, 1.0]);
            assert!(p.iter().all(|&v| v > 0.0 && v < 1.0), "{p:?}");
            e.reward(t % 2, if t % 2 == 0 { 1.0 } else { -1.0 })
                .unwrap();
            let mut sums = vec![0.0; e.output_ids.len()];
            for edge in &e.plastic_edges {
                sums[edge.post] += edge.weight.abs();
            }
            assert!(sums.iter().all(|&v| v <= 1.000001), "{sums:?}");
        }
    }
    #[test]
    fn negative_readout_gain_gradient_matches_forward_finite_difference() {
        let mut e = fixture();
        e.set_output_gains(&[-1.0, 1.0]).unwrap();
        e.plastic_edges[0].weight = 0.5;
        let before = e.plastic_edges[0].weight;
        let eps = 1e-3;
        e.plastic_edges[0].weight = before + eps;
        let plus = e.forward(&[1.0, 0.0])[0].ln();
        e.plastic_edges[0].weight = before - eps;
        let minus = e.forward(&[1.0, 0.0])[0].ln();
        let numeric = (plus - minus) / (2.0 * eps);
        e.plastic_edges[0].weight = before;
        e.forward(&[1.0, 0.0]);
        e.reward(0, 1.0).unwrap();
        let analytic = (e.plastic_edges[0].weight - before) / e.config.learning_rate;
        assert!(analytic < 0.0);
        assert!((numeric - analytic).abs() < 0.001, "{numeric} {analytic}");
    }
    #[test]
    fn gain_validation_is_atomic_and_checkpoint_roundtrips() {
        let mut e = fixture();
        for invalid in [[0.0, 1.0], [f32::NAN, 1.0], [1.1, 1.0]] {
            assert!(e.set_output_gains(&invalid).is_err());
            assert_eq!(e.output_gains(), &[1.0, 1.0]);
        }
        assert!(e.set_output_gains(&[1.0]).is_err());
        e.set_output_gains(&[-1.0, 0.5]).unwrap();
        let p = path("gains");
        e.save_checkpoint(&p).unwrap();
        let restored = Engine::load_checkpoint(&p).unwrap();
        assert_eq!(restored.output_gains(), &[-1.0, 0.5]);
        fs::remove_file(p).unwrap();
    }
    #[test]
    fn schema_three_checkpoint_defaults_to_unit_gains() {
        use safetensors::tensor::serialize_to_file;
        let e = fixture();
        let modern = path("modern");
        let legacy = path("legacy");
        e.save_checkpoint(&modern).unwrap();
        let bytes = fs::read(&modern).unwrap();
        let (_, header) = safetensors::SafeTensors::read_metadata(&bytes).unwrap();
        let mut metadata = header.metadata().as_ref().unwrap().clone();
        metadata.insert("version".into(), "3".into());
        let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
        let views: Vec<_> = st
            .names()
            .iter()
            .filter(|&&name| name != "output_gains")
            .map(|&name| (name, st.tensor(name).unwrap()))
            .collect();
        serialize_to_file(views, Some(metadata), &legacy).unwrap();
        let restored = Engine::load_checkpoint(&legacy).unwrap();
        assert_eq!(restored.output_gains(), &[1.0, 1.0]);
        fs::remove_file(modern).unwrap();
        fs::remove_file(legacy).unwrap();
    }
    #[test]
    fn extreme_finite_inputs_do_not_overflow() {
        let mut e = fixture();
        let p = e.forward(&[f32::MAX, f32::MAX]);
        assert!(p.iter().all(|v| v.is_finite()));
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(e.hidden_activity().iter().all(|v| v.is_finite()));
    }
    #[test]
    fn checkpoint_rejects_excess_homeostatic_mass() {
        let mut e = fixture();
        e.config.homeostasis = true;
        e.plastic_edges[0].weight = 0.8;
        e.plastic_edges[2].sign = 1;
        e.plastic_edges[2].weight = 0.8;
        let p = path("bad_mass");
        e.save_checkpoint(&p).unwrap();
        let error = Engine::load_checkpoint(&p).err().unwrap();
        assert!(error.contains("L1 mass"), "{error}");
        fs::remove_file(p).unwrap();
    }
    #[test]
    fn malformed_edges_rejected_and_input_silent_hidden_retained() {
        let a = path("bad_i");
        let b = path("bad_p");
        fs::write(&a, "1 10 1 2\n").unwrap();
        fs::write(&b, "10 100 1 1\n").unwrap();
        assert!(Engine::load(&a, &b, PlasticConfig::default()).is_err());
        fs::write(&a, "1 10 1 1\n").unwrap();
        fs::write(&b, "11 100 1 1\n10 200 1 1\n").unwrap();
        let mut e = Engine::load(&a, &b, PlasticConfig::default()).unwrap();
        assert_eq!(e.hidden_ids(), &[10, 11]);
        e.forward(&[1.0]);
        assert_eq!(e.hidden_activity()[1], 0.0);
        assert!(e.save_checkpoint(path("pending")).is_err());
        e.clear_pending();
        let old = e.output_actions().to_vec();
        assert!(e.set_output_actions(&[0, 0]).is_err());
        assert_eq!(e.output_actions(), old);
        let before: Vec<_> = e.weights().collect();
        e.forward(&[1.0]);
        e.reward(0, 0.0).unwrap();
        assert_eq!(e.weights().collect::<Vec<_>>(), before);
        fs::remove_file(a).unwrap();
        fs::remove_file(b).unwrap();
    }
}
