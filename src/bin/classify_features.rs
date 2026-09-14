//! Batch inference for generic, precomputed features.
use nobi::plastic::Engine;
use safetensors::{tensor::Dtype, SafeTensors};
use std::{
    env,
    error::Error,
    fs,
    io::{self, BufWriter, Write},
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

fn run() -> Result<()> {
    let args: Vec<_> = env::args_os().collect();
    if args.len() != 3 {
        return Err("usage: classify_features CHECKPOINT FEATURES.safetensors".into());
    }
    let mut engine = Engine::load_checkpoint(&args[1])?;
    let bytes = fs::read(&args[2])?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    let ids = tensors.tensor("input_ids")?;
    let dimension = engine.input_ids().len();
    if ids.dtype() != Dtype::U64
        || ids.shape() != [dimension]
        || ids
            .data()
            .chunks_exact(8)
            .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
            .ne(engine.input_ids().iter().copied())
    {
        return Err("feature input_ids mismatch checkpoint input order".into());
    }
    let input = tensors.tensor("inputs")?;
    if input.dtype() != Dtype::F32
        || input.shape().len() != 2
        || input.shape()[0] == 0
        || input.shape()[1] != dimension
    {
        return Err("inputs must be nonempty F32[N,D] matching checkpoint input dimension".into());
    }
    let features: Vec<f32> = input
        .data()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    if features.iter().any(|&x| !x.is_finite() || x < 0.0) {
        return Err("inputs must be finite and nonnegative".into());
    }
    let weights = engine.weights().collect::<Vec<_>>();
    let mut out = BufWriter::new(io::stdout().lock());
    write!(out, "index\tpredicted")?;
    for j in 0..engine.config().actions {
        write!(out, "\tp{j}")?;
    }
    writeln!(out)?;
    for (i, row) in features.chunks_exact(dimension).enumerate() {
        let p = engine.forward(row);
        let mut best = 0;
        for j in 1..p.len() {
            if p[j] > p[best] {
                best = j;
            }
        }
        write!(out, "{i}\t{best}")?;
        for &v in p {
            write!(out, "\t{v:.9}")?;
        }
        writeln!(out)?;
        engine.clear_pending();
    }
    out.flush()?;
    if !engine.weights().zip(weights).all(|(a, b)| a == b) {
        return Err("inference changed model weights".into());
    }
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("classify_features: {e}");
        std::process::exit(1);
    }
}
