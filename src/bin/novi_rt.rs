//! Soft periodic runner for one fixed sparse rate vector and a saved model.
use novi::plastic::Engine;
use novi::runtime::Runtime;
use std::collections::BTreeMap;
use std::env;
use std::error::Error;
use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};

type R<T> = Result<T, Box<dyn Error>>;

fn sparse_input(path: &PathBuf, ids: &[u64]) -> R<Vec<f32>> {
    parse_sparse_input(&fs::read_to_string(path)?, ids)
}

fn parse_sparse_input(contents: &str, ids: &[u64]) -> R<Vec<f32>> {
    let positions: BTreeMap<u64, usize> = ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
    let mut values = vec![0.0; ids.len()];
    let mut seen = vec![false; ids.len()];
    let lines: Vec<_> = contents
        .lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty() && !line.trim_start().starts_with('#'))
        .collect();
    if lines.is_empty() {
        return Err("input must contain one or more port rates".into());
    }
    let colon_form = lines[0].1.contains(':');
    if colon_form && lines.len() != 1 {
        return Err("colon-form input must be exactly one row".into());
    }
    for (line_number, line) in lines {
        let fields: Vec<_> = line.split_whitespace().collect();
        if !colon_form && fields.len() != 2 {
            return Err(format!("line {}: expected port_id rate", line_number + 1).into());
        }
        for field in if colon_form {
            fields.chunks_exact(1)
        } else {
            fields.chunks_exact(2)
        } {
            let (id_text, rate_text) = if colon_form {
                field[0].split_once(':').ok_or("expected port_id:rate")?
            } else {
                (field[0], field[1])
            };
            let id: u64 = id_text.parse()?;
            let rate: f32 = rate_text.parse()?;
            let &position = positions
                .get(&id)
                .ok_or_else(|| format!("unknown input port ID {id}"))?;
            if !rate.is_finite() || rate < 0.0 || seen[position] {
                return Err(
                    format!("line {}: invalid or duplicate port rate", line_number + 1).into(),
                );
            }
            values[position] = rate;
            seen[position] = true;
        }
    }
    Ok(values)
}

/// Advance to the next future slot, counting dropped periods instead of catching up.
fn next_slot(scheduled_start: Instant, finish: Instant, period: Duration) -> R<(Instant, u64)> {
    let mut next = scheduled_start
        .checked_add(period)
        .ok_or("scheduler instant overflow")?;
    if finish <= next {
        return Ok((next, 0));
    }
    let period_ns = period.as_nanos();
    let skipped = (finish.duration_since(next).as_nanos() / period_ns) + 1;
    let skipped_u64 = u64::try_from(skipped)?;
    let advance = period
        .checked_mul(u32::try_from(skipped)?)
        .ok_or("scheduler duration overflow")?;
    next = next
        .checked_add(advance)
        .ok_or("scheduler instant overflow")?;
    Ok((next, skipped_u64))
}

fn run() -> R<()> {
    let args: Vec<_> = env::args().collect();
    if args.len() != 7 || args[3] != "--ticks" || args[5] != "--period-us" {
        return Err("usage: novi_rt CHECKPOINT INPUT.tsv --ticks N --period-us N".into());
    }
    let ticks: u64 = args[4].parse()?;
    let period_us: u64 = args[6].parse()?;
    if ticks == 0 || period_us == 0 {
        return Err("ticks and period-us must be positive".into());
    }
    let period = Duration::from_micros(period_us);
    let engine = Engine::load_checkpoint(&args[1])?;
    let input = sparse_input(&PathBuf::from(&args[2]), engine.input_ids())?;
    let mut runtime = Runtime::new(engine, period)?;
    let begin = Instant::now();
    let mut scheduled_start = begin;
    let mut scheduled_finish_misses = 0u64;
    let mut late_starts = 0u64;
    let mut skipped_periods = 0u64;
    let mut max_compute_ns = 0u128;
    let mut total_compute_ns = 0u128;
    let mut action = 0usize;
    for tick in 0..ticks {
        let actual_start = Instant::now();
        if actual_start > scheduled_start {
            late_starts += 1;
        }
        let report = runtime.tick(&input, None)?;
        action = report.selected_action;
        let compute_ns = report.compute_time.as_nanos();
        max_compute_ns = max_compute_ns.max(compute_ns);
        total_compute_ns += compute_ns;
        let finish = Instant::now();
        let deadline = scheduled_start
            .checked_add(period)
            .ok_or("scheduler instant overflow")?;
        if finish > deadline {
            scheduled_finish_misses += 1;
        }
        if tick + 1 < ticks {
            let (next, skipped) = next_slot(scheduled_start, finish, period)?;
            skipped_periods += skipped;
            if let Some(rest) = next.checked_duration_since(Instant::now()) {
                std::thread::sleep(rest);
            }
            scheduled_start = next;
        }
    }
    println!("{{\"mode\":\"soft-real-time\",\"ticks\":{ticks},\"period_us\":{period_us},\"compute_deadline_misses\":{},\"scheduled_finish_misses\":{scheduled_finish_misses},\"late_starts\":{late_starts},\"skipped_periods\":{skipped_periods},\"mean_compute_ns\":{},\"max_compute_ns\":{max_compute_ns},\"elapsed_ms\":{},\"selected_action\":{action}}}",runtime.compute_deadline_misses(),total_compute_ns/ticks as u128,begin.elapsed().as_millis());
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("novi_rt: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sparse_input_accepts_both_forms_and_rejects_bad_ids() {
        let ids = [1u64, 2, 3];
        assert_eq!(
            parse_sparse_input("1 0.5\n3 1\n", &ids).unwrap(),
            vec![0.5, 0.0, 1.0]
        );
        assert_eq!(
            parse_sparse_input("1:0.5\t3:1\n", &ids).unwrap(),
            vec![0.5, 0.0, 1.0]
        );
        assert!(parse_sparse_input("4:1\n", &ids).is_err());
        assert!(parse_sparse_input("1:1\t1:2\n", &ids).is_err());
        assert!(parse_sparse_input("1 1\n1 2\n", &ids).is_err());
    }
    #[test]
    fn scheduler_drops_overdue_slots() {
        let start = Instant::now();
        let period = Duration::from_millis(10);
        assert_eq!(
            next_slot(start, start + Duration::from_millis(9), period)
                .unwrap()
                .1,
            0
        );
        let (next, skipped) = next_slot(start, start + Duration::from_millis(35), period).unwrap();
        assert_eq!(skipped, 3);
        assert_eq!(next.duration_since(start), Duration::from_millis(40));
    }
}
