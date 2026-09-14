use nobi::Agent;
use std::{error::Error, time::Instant};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str).unwrap_or("train") {
        "train" | "bench" => {
            let bench = args.get(1).is_some_and(|s| s == "bench");
            let episodes: usize = args
                .get(2)
                .map(|s| s.parse())
                .transpose()?
                .unwrap_or(if bench { 1_000_000 } else { 300 });
            if episodes == 0 {
                return Err("episodes must be positive".into());
            }
            let mut agent = Agent::new(42);
            let before = (0..2)
                .filter(|&cue| agent.action(cue, false) == cue)
                .count();
            let start = Instant::now();
            let mut correct = 0usize;
            for episode in 0..episodes {
                let cue = episode % 2;
                let action = std::hint::black_box(agent.action(cue, true));
                correct += usize::from(action == cue);
                agent.reward(if action == cue { 1.0 } else { -1.0 });
            }
            let elapsed = start.elapsed().as_secs_f64();
            let after = (0..2)
                .filter(|&cue| agent.action(cue, false) == cue)
                .count();
            println!(
                "synthetic cue learning: greedy accuracy {}% -> {}%; training {correct}/{episodes}",
                before * 50,
                after * 50
            );
            println!(
                "{episodes} episodes in {elapsed:.6}s ({:.0} episodes/s)",
                episodes as f64 / elapsed
            );
        }
        "connectome" => {
            let path = args
                .get(2)
                .ok_or("usage: nobi connectome edges.tsv [steps] [root_id]")?;
            let steps: usize = args.get(3).map(|s| s.parse()).transpose()?.unwrap_or(1000);
            if steps == 0 {
                return Err("steps must be positive".into());
            }
            let start = Instant::now();
            let graph = nobi::connectome::Graph::load(path)?;
            let root: u64 = args
                .get(4)
                .map(|s| s.parse())
                .transpose()?
                .unwrap_or(*graph.ids().first().ok_or("empty graph")?);
            println!(
                "loaded {} neurons, {} edges in {:.3}s",
                graph.neuron_count(),
                graph.edge_count(),
                start.elapsed().as_secs_f64()
            );
            let mut sim = nobi::connectome::Simulator::new(graph);
            let start = Instant::now();
            let mut spikes = 0u64;
            for _ in 0..steps {
                sim.stimulate_by_id(root, 1.5)?;
                spikes += sim.step() as u64;
            }
            let elapsed = start.elapsed().as_secs_f64();
            println!(
                "root_id={root}, steps={steps}, spikes={spikes}, propagated_edges={}",
                sim.total_synaptic_events()
            );
            println!(
                "simulation {elapsed:.6}s, {:.0} neuron updates/s",
                steps as f64 * sim.neuron_count() as f64 / elapsed
            );
        }
        _ => return Err(
            "commands: train [episodes], bench [episodes], connectome edges.tsv [steps] [root_id]"
                .into(),
        ),
    }
    Ok(())
}
