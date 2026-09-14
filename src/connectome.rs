//! Sparse connectome import and a deterministic, fixed-step LIF-like simulator.
//!
//! Input is headerless TSV: `pre_id<TAB>post_id<TAB>syn_count<TAB>sign`.
//! Counts are positive integers; signs are -1, 0, or 1. For each
//! source neuron, edge weights are `sign * count / sum(abs(sign * count))`.
//! This outgoing normalization is an engineering choice to keep the numerical
//! scale bounded; it is not a claim of biological synaptic calibration.

use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Debug)]
pub enum ConnectomeError {
    Io(std::io::Error),
    InvalidLine { line: usize, reason: String },
    TooManyNeurons,
    UnknownNeuron(u64),
    InvalidCurrent,
}

impl fmt::Display for ConnectomeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "connectome I/O error: {err}"),
            Self::InvalidLine { line, reason } => write!(f, "connectome line {line}: {reason}"),
            Self::TooManyNeurons => write!(f, "connectome has more than u32::MAX neurons"),
            Self::UnknownNeuron(id) => write!(f, "unknown neuron ID {id}"),
            Self::InvalidCurrent => write!(f, "stimulus current must be finite"),
        }
    }
}
impl Error for ConnectomeError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            _ => None,
        }
    }
}
impl From<std::io::Error> for ConnectomeError {
    fn from(err: std::io::Error) -> Self {
        Self::Io(err)
    }
}

/// CSR adjacency with sorted public FlyWire IDs and dense `u32` indices.
#[derive(Debug)]
pub struct Graph {
    ids: Vec<u64>,
    offsets: Vec<usize>,
    targets: Vec<u32>,
    weights: Vec<f32>,
    index: HashMap<u64, u32>,
}

impl Graph {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, ConnectomeError> {
        Self::from_tsv(path)
    }

    pub fn from_tsv(path: impl AsRef<Path>) -> Result<Self, ConnectomeError> {
        let mut reader = BufReader::new(File::open(path)?);
        let mut records: Vec<(u64, u64, f64)> = Vec::new();
        let mut unique_ids = HashSet::new();
        let mut line = String::new();
        let mut line_no = 0;
        loop {
            line.clear();
            if reader.read_line(&mut line)? == 0 {
                break;
            }
            line_no += 1;
            let row = line.strip_suffix('\n').unwrap_or(&line);
            let row = row.strip_suffix('\r').unwrap_or(row);
            let mut columns = row.split('\t');
            let invalid = |reason: &str| ConnectomeError::InvalidLine {
                line: line_no,
                reason: reason.to_owned(),
            };
            let (Some(pre_field), Some(post_field), Some(count_field), Some(sign_field), None) = (
                columns.next(),
                columns.next(),
                columns.next(),
                columns.next(),
                columns.next(),
            ) else {
                return Err(invalid("expected exactly four tab-separated fields"));
            };
            let pre = pre_field
                .parse::<u64>()
                .map_err(|_| invalid("invalid pre_id"))?;
            let post = post_field
                .parse::<u64>()
                .map_err(|_| invalid("invalid post_id"))?;
            if pre == 0 || post == 0 {
                return Err(invalid("neuron IDs must be nonzero"));
            }
            let count = count_field
                .parse::<u64>()
                .map_err(|_| invalid("invalid syn_count"))?;
            if count == 0 {
                return Err(invalid("syn_count must be positive"));
            }
            let sign = sign_field
                .parse::<f64>()
                .map_err(|_| invalid("invalid sign"))?;
            if !matches!(sign, -1.0 | 0.0 | 1.0) {
                return Err(invalid("sign must be -1, 0, or 1"));
            }
            records.push((pre, post, count as f64 * sign));
            unique_ids.insert(pre);
            unique_ids.insert(post);
        }
        let mut ids: Vec<u64> = unique_ids.into_iter().collect();
        ids.sort_unstable();
        if ids.len() > u32::MAX as usize {
            return Err(ConnectomeError::TooManyNeurons);
        }
        let index: HashMap<u64, u32> = ids
            .iter()
            .enumerate()
            .map(|(i, id)| (*id, i as u32))
            .collect();
        records.sort_unstable_by_key(|&(pre, post, _)| (pre, post));
        let mut offsets = Vec::with_capacity(ids.len() + 1);
        let mut targets = Vec::with_capacity(records.len());
        let mut weights = Vec::with_capacity(records.len());
        offsets.push(0);
        let mut cursor = 0;
        for &id in &ids {
            let start = cursor;
            while cursor < records.len() && records[cursor].0 == id {
                targets.push(index[&records[cursor].1]);
                weights.push(records[cursor].2 as f32);
                cursor += 1;
            }
            // Normalize by outgoing absolute strength, including zero-sign edges.
            // Accumulate in f64 before converting to f32 for stable large counts.
            let sum: f64 = records[start..cursor].iter().map(|r| r.2.abs()).sum();
            if sum > 0.0 {
                for (record, weight) in records[start..cursor]
                    .iter()
                    .zip(&mut weights[start..cursor])
                {
                    *weight = (record.2 / sum) as f32;
                }
            }
            offsets.push(cursor);
        }
        Ok(Self {
            ids,
            offsets,
            targets,
            weights,
            index,
        })
    }

    pub fn neuron_count(&self) -> usize {
        self.ids.len()
    }
    pub fn edge_count(&self) -> usize {
        self.targets.len()
    }
    pub fn ids(&self) -> &[u64] {
        &self.ids
    }
    pub fn index_of(&self, id: u64) -> Option<u32> {
        self.index.get(&id).copied()
    }
    pub fn outgoing(&self, index: u32) -> Option<impl Iterator<Item = (u32, f32)> + '_> {
        let i = index as usize;
        if i >= self.ids.len() {
            return None;
        }
        let range = self.offsets[i]..self.offsets[i + 1];
        Some(
            self.targets[range.clone()]
                .iter()
                .copied()
                .zip(self.weights[range].iter().copied()),
        )
    }
}

/// Synchronous fixed-step leaky integrate-and-fire approximation.
///
/// A spike at tick N contributes to its targets at tick N+1. Voltage leaks by
/// 10% per tick, threshold is 1, and a spiking cell resets to 0. Stimuli are
/// one-tick currents. No allocation occurs during `step`.
pub struct Simulator {
    graph: Graph,
    voltage: Vec<f32>,
    stimulus: Vec<f32>,
    incoming: Vec<f32>,
    next_incoming: Vec<f32>,
    spikes: Vec<u32>,
    total_synaptic_events: u64,
}

impl Simulator {
    pub fn new(graph: Graph) -> Self {
        let n = graph.neuron_count();
        Self {
            graph,
            voltage: vec![0.0; n],
            stimulus: vec![0.0; n],
            incoming: vec![0.0; n],
            next_incoming: vec![0.0; n],
            spikes: Vec::with_capacity(n),
            total_synaptic_events: 0,
        }
    }
    pub fn neuron_count(&self) -> usize {
        self.graph.neuron_count()
    }
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }
    pub fn graph(&self) -> &Graph {
        &self.graph
    }
    pub fn total_synaptic_events(&self) -> u64 {
        self.total_synaptic_events
    }
    pub fn voltage_by_id(&self, id: u64) -> Option<f32> {
        self.graph.index_of(id).map(|i| self.voltage[i as usize])
    }
    pub fn stimulate_by_id(&mut self, id: u64, current: f32) -> Result<(), ConnectomeError> {
        if !current.is_finite() {
            return Err(ConnectomeError::InvalidCurrent);
        }
        let i = self
            .graph
            .index_of(id)
            .ok_or(ConnectomeError::UnknownNeuron(id))? as usize;
        let sum = self.stimulus[i] + current;
        if !sum.is_finite() {
            return Err(ConnectomeError::InvalidCurrent);
        }
        self.stimulus[i] = sum;
        Ok(())
    }
    pub fn step(&mut self) -> usize {
        self.spikes.clear();
        for i in 0..self.voltage.len() {
            let v = self.voltage[i] * 0.9 + self.incoming[i] + self.stimulus[i];
            self.stimulus[i] = 0.0;
            self.incoming[i] = 0.0;
            if v >= 1.0 {
                self.voltage[i] = 0.0;
                self.spikes.push(i as u32);
            } else {
                self.voltage[i] = v;
            }
        }
        for &source in &self.spikes {
            let range =
                self.graph.offsets[source as usize]..self.graph.offsets[source as usize + 1];
            for edge in range {
                self.next_incoming[self.graph.targets[edge] as usize] += self.graph.weights[edge];
                self.total_synaptic_events = self.total_synaptic_events.saturating_add(1);
            }
        }
        std::mem::swap(&mut self.incoming, &mut self.next_incoming);
        self.spikes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    fn graph(input: &str) -> Result<Graph, ConnectomeError> {
        let path = std::env::temp_dir().join(format!(
            "novi-connectome-{}-{:?}.tsv",
            std::process::id(),
            std::thread::current().id()
        ));
        let mut file = File::create(&path).unwrap();
        file.write_all(input.as_bytes()).unwrap();
        drop(file);
        let result = Graph::from_tsv(&path);
        std::fs::remove_file(path).unwrap();
        result
    }
    #[test]
    fn normalized_polarity_and_sorted_ids() {
        let g = graph("30\t20\t1\t-1\n30\t10\t3\t1\n").unwrap();
        assert_eq!(g.ids(), &[10, 20, 30]);
        assert_eq!(g.edge_count(), 2);
        let edges: Vec<_> = g.outgoing(g.index_of(30).unwrap()).unwrap().collect();
        assert_eq!(edges, vec![(0, 0.75), (1, -0.25)]);
    }
    #[test]
    fn one_tick_delay_and_determinism() {
        let input = "1\t2\t1\t1\n2\t3\t1\t-1\n";
        let mut a = Simulator::new(graph(input).unwrap());
        let mut b = Simulator::new(graph(input).unwrap());
        for sim in [&mut a, &mut b] {
            sim.stimulate_by_id(1, 1.0).unwrap();
            assert_eq!(sim.step(), 1);
            assert_eq!(sim.voltage_by_id(2), Some(0.0));
            assert_eq!(sim.step(), 1);
            assert_eq!(sim.step(), 0);
            assert_eq!(sim.voltage_by_id(3), Some(-1.0));
            assert_eq!(sim.total_synaptic_events(), 2);
        }
    }
    #[test]
    fn invalid_data_rejected() {
        for input in [
            "a\t2\t1\t1\n",
            "0\t2\t1\t1\n",
            "1\t2\t0\t1\n",
            "1\t2\t-2\t1\n",
            "1\t2\t1\tNaN\n",
            "1\t2\t1\t2\n",
            "1\t2\t1\t0.5\n",
            "1\t2\t1\n",
        ] {
            assert!(graph(input).is_err(), "accepted {input:?}");
        }
    }
}
