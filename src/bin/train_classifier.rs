use novi::plastic::{Engine, PlasticConfig};
use safetensors::{tensor::Dtype, SafeTensors};
use std::{
    env,
    error::Error,
    fs::{self, File},
    io::{BufWriter, Write},
    path::PathBuf,
    time::Instant,
};
type R<T> = Result<T, Box<dyn Error>>;
struct Split {
    x: Vec<f32>,
    y: Vec<usize>,
    d: usize,
}
impl Split {
    fn len(&self) -> usize {
        self.y.len()
    }
    fn row(&self, i: usize) -> &[f32] {
        &self.x[i * self.d..(i + 1) * self.d]
    }
}
fn split(st: &SafeTensors<'_>, name: &str, d: usize, k: usize) -> R<Split> {
    let x = st.tensor(&format!("{name}_inputs"))?;
    let y = st.tensor(&format!("{name}_labels"))?;
    if x.dtype() != Dtype::F32
        || x.shape().len() != 2
        || x.shape()[1] != d
        || x.shape()[0] == 0
        || y.dtype() != Dtype::I64
        || y.shape() != [x.shape()[0]]
    {
        return Err(format!("invalid {name} tensors").into());
    }
    let x: Vec<f32> = x
        .data()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    if x.iter().any(|v| !v.is_finite() || *v < 0.) {
        return Err(format!("{name} features must be finite nonnegative").into());
    }
    let mut labels = Vec::with_capacity(y.shape()[0]);
    for b in y.data().chunks_exact(8) {
        let v = i64::from_le_bytes(b.try_into().unwrap());
        if v < 0 || v as u64 >= k as u64 {
            return Err(format!("{name} label out of range: {v}").into());
        }
        labels.push(v as usize)
    }
    Ok(Split { x, y: labels, d })
}
struct Rng(u64);
impl Rng {
    fn new(x: u64) -> Self {
        Self(x.max(1))
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn shuffle<T>(&mut self, x: &mut [T]) {
        for i in (1..x.len()).rev() {
            x.swap(i, (self.next() % (i as u64 + 1)) as usize)
        }
    }
    fn sample(&mut self, p: &[f32]) -> usize {
        let mut u = (self.next() >> 40) as f32 / 16_777_216.;
        for (i, &v) in p.iter().enumerate() {
            u -= v;
            if u < 0. {
                return i;
            }
        }
        p.len() - 1
    }
}
#[derive(Clone, Copy, Default)]
struct Metric {
    acc: f64,
    ce: f64,
    prob: f64,
}
fn evaluate(e: &mut Engine, s: &Split, mut out: Option<&mut dyn Write>) -> R<Metric> {
    let before = e.weights().collect::<Vec<_>>();
    let mut m = Metric::default();
    for i in 0..s.len() {
        let p = e.forward(s.row(i));
        let y = s.y[i];
        let mut best = 0;
        for j in 1..p.len() {
            if p[j] > p[best] {
                best = j
            }
        }
        m.acc += f64::from(best == y);
        m.prob += p[y] as f64;
        m.ce -= (p[y].max(1e-30) as f64).ln();
        if let Some(w) = out.as_deref_mut() {
            write!(w, "{i}\t{y}\t{best}")?;
            for &v in p {
                write!(w, "\t{v:.9}")?
            }
            writeln!(w)?
        }
        e.clear_pending()
    }
    let n = s.len() as f64;
    m.acc /= n;
    m.ce /= n;
    m.prob /= n;
    if !e.weights().zip(before).all(|(a, b)| a == b) {
        return Err("evaluation changed weights".into());
    }
    Ok(m)
}
#[derive(Clone, Copy)]
enum Mode {
    Supervised,
    Reward,
    Shuffled,
}
impl Mode {
    fn name(self) -> &'static str {
        match self {
            Self::Supervised => "supervised",
            Self::Reward => "reward",
            Self::Shuffled => "shuffled",
        }
    }
}
#[derive(Clone, Copy)]
enum Phase {
    Tune,
    Full,
}
impl Phase {
    fn name(self) -> &'static str {
        match self {
            Self::Tune => "tune",
            Self::Full => "full",
        }
    }
}
#[derive(Clone, Copy)]
enum Readout {
    Positive,
    Opponent,
}
impl Readout {
    fn name(self) -> &'static str {
        match self {
            Self::Positive => "positive",
            Self::Opponent => "opponent",
        }
    }
}
fn configure_readout(engine: &mut Engine, readout: Readout) -> R<()> {
    if matches!(readout, Readout::Opponent) {
        let actions = engine.config().actions;
        let gains: Vec<f32> = (0..engine.output_ids().len())
            .map(|i| {
                if (i / actions).is_multiple_of(2) {
                    1.0
                } else {
                    -1.0
                }
            })
            .collect();
        engine.set_output_gains(&gains)?;
    }
    Ok(())
}
struct Args {
    features: PathBuf,
    input: PathBuf,
    plastic: PathBuf,
    out: PathBuf,
    epochs: usize,
    seed: u64,
    lr: f32,
    gain: f32,
    active: f32,
    homeostasis: bool,
    batch_size: usize,
    mode: Mode,
    actions: usize,
    patience: usize,
    phase: Phase,
    readout: Readout,
}
fn args() -> R<Args> {
    let a: Vec<String> = env::args().collect();
    if a.len() < 5 {
        return Err("usage: train_classifier FEATURES INPUT_EDGES PLASTIC_EDGES OUTDIR [--epochs N --seed N --lr FLOAT --gain FLOAT --active-fraction FLOAT --homeostasis true|false --batch-size N --mode supervised|reward|shuffled --actions N --patience N --phase tune|full --readout positive|opponent]".into());
    }
    let mut o = Args {
        features: a[1].clone().into(),
        input: a[2].clone().into(),
        plastic: a[3].clone().into(),
        out: a[4].clone().into(),
        epochs: 20,
        seed: 1,
        lr: 0.001,
        gain: 12.,
        active: 0.1,
        homeostasis: true,
        batch_size: 1,
        mode: Mode::Supervised,
        actions: 10,
        patience: 5,
        phase: Phase::Full,
        readout: Readout::Positive,
    };
    if !(a.len() - 5).is_multiple_of(2) {
        return Err("flag missing value".into());
    }
    for p in a[5..].chunks_exact(2) {
        match p[0].as_str() {
            "--epochs" => o.epochs = p[1].parse()?,
            "--seed" => o.seed = p[1].parse()?,
            "--lr" => o.lr = p[1].parse()?,
            "--gain" => o.gain = p[1].parse()?,
            "--active-fraction" => o.active = p[1].parse()?,
            "--homeostasis" => o.homeostasis = p[1].parse()?,
            "--batch-size" => o.batch_size = p[1].parse()?,
            "--actions" => o.actions = p[1].parse()?,
            "--patience" => o.patience = p[1].parse()?,
            "--readout" => {
                o.readout = match p[1].as_str() {
                    "positive" => Readout::Positive,
                    "opponent" => Readout::Opponent,
                    _ => return Err("readout must be positive or opponent".into()),
                }
            }
            "--phase" => {
                o.phase = match p[1].as_str() {
                    "tune" => Phase::Tune,
                    "full" => Phase::Full,
                    _ => return Err("phase must be tune or full".into()),
                }
            }
            "--mode" => {
                o.mode = match p[1].as_str() {
                    "supervised" => Mode::Supervised,
                    "reward" => Mode::Reward,
                    "shuffled" => Mode::Shuffled,
                    _ => return Err("invalid mode".into()),
                }
            }
            _ => return Err(format!("unknown flag {}", p[0]).into()),
        }
    }
    if o.epochs == 0 || o.patience == 0 || o.batch_size == 0 || o.actions < 2 {
        return Err("epochs, patience, and batch size must be positive; actions >= 2".into());
    }
    Ok(o)
}
fn run() -> R<()> {
    let a = args()?;
    let start = Instant::now();
    fs::create_dir_all(&a.out)?;
    if matches!(a.phase, Phase::Tune) {
        let stale_predictions = a.out.join("predictions.tsv");
        if stale_predictions.exists() {
            fs::remove_file(stale_predictions)?;
        }
    }
    let cfg = PlasticConfig {
        actions: a.actions,
        learning_rate: a.lr,
        logit_gain: a.gain,
        active_fraction: a.active,
        homeostasis: a.homeostasis,
        ..PlasticConfig::default()
    };
    let mut e = Engine::load(&a.input, &a.plastic, cfg)?;
    configure_readout(&mut e, a.readout)?;
    let bytes = fs::read(&a.features)?;
    let st = SafeTensors::deserialize(&bytes)?;
    let ids = st.tensor("input_ids")?;
    if ids.dtype() != Dtype::U64
        || ids.shape() != [e.input_ids().len()]
        || ids
            .data()
            .chunks_exact(8)
            .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
            .ne(e.input_ids().iter().copied())
    {
        return Err("feature input_ids mismatch engine input order".into());
    }
    let d = e.input_ids().len();
    let train = split(&st, "train", d, a.actions)?;
    let val = split(&st, "validation", d, a.actions)?;
    let test = split(&st, "test", d, a.actions)?;
    let initial = e.weights().collect::<Vec<_>>();
    let baseline = evaluate(&mut e, &val, None)?;
    let model = a.out.join("model.safetensors");
    e.save_checkpoint(&model)?;
    let mut best = baseline;
    let mut best_epoch = 0;
    let mut stale = 0;
    let mut order: Vec<usize> = (0..train.len()).collect();
    let mut sr = Rng::new(a.seed ^ 0x9e37_79b9_7f4a_7c15);
    let mut dr = Rng::new(a.seed ^ 0xd1b5_4a32_d192_ed03);
    let mut labels = train.y.clone();
    if matches!(a.mode, Mode::Shuffled) {
        Rng::new(a.seed ^ 0xa24b_aed4_963e_e407).shuffle(&mut labels)
    }
    let mut curves = BufWriter::new(File::create(a.out.join("curves.csv"))?);
    writeln!(curves,"epoch,train_cross_entropy,validation_accuracy,validation_cross_entropy,validation_correct_probability,elapsed_seconds")?;
    writeln!(
        curves,
        "0,,{:.9},{:.9},{:.9},{:.3}",
        baseline.acc,
        baseline.ce,
        baseline.prob,
        start.elapsed().as_secs_f64()
    )?;
    for epoch in 1..=a.epochs {
        sr.shuffle(&mut order);
        let mut ce = 0.;
        let mut accumulated = 0usize;
        for &i in &order {
            let y = labels[i];
            let p = e.forward(train.row(i));
            ce -= (p[y].max(1e-30) as f64).ln();
            let (action, reward) = match a.mode {
                Mode::Reward => {
                    let action = dr.sample(p);
                    (action, if action == y { 1.0 } else { -1.0 })
                }
                _ => (y, 1.0),
            };
            if a.batch_size == 1 {
                e.reward(action, reward)?;
            } else {
                e.accumulate_reward(action, reward)?;
                accumulated += 1;
                if accumulated == a.batch_size {
                    e.apply_batch()?;
                    accumulated = 0;
                }
            }
        }
        if accumulated > 0 {
            e.apply_batch()?;
        }
        let m = evaluate(&mut e, &val, None)?;
        writeln!(
            curves,
            "{epoch},{:.9},{:.9},{:.9},{:.9},{:.3}",
            ce / train.len() as f64,
            m.acc,
            m.ce,
            m.prob,
            start.elapsed().as_secs_f64()
        )?;
        println!(
            "epoch {epoch}: train_ce={:.5} val_acc={:.4} val_ce={:.5}",
            ce / train.len() as f64,
            m.acc,
            m.ce
        );
        if m.acc > best.acc || (m.acc == best.acc && m.ce < best.ce - 1e-12) {
            best = m;
            best_epoch = epoch;
            stale = 0;
            e.save_checkpoint(&model)?
        } else {
            stale += 1;
            if stale >= a.patience {
                break;
            }
        }
    }
    curves.flush()?;
    let mut selected = Engine::load_checkpoint(&model)?;
    let restored = evaluate(&mut selected, &val, None)?;
    if (restored.acc - best.acc).abs() > 1e-12 || (restored.ce - best.ce).abs() > 1e-8 {
        return Err("checkpoint validation mismatch".into());
    }
    let final_weights = selected.weights().collect::<Vec<_>>();
    let changed = initial
        .iter()
        .zip(&final_weights)
        .filter(|(x, y)| x != y)
        .count();
    let delta: f64 = initial
        .iter()
        .zip(&final_weights)
        .map(|(x, y)| f64::from((x - y).abs()))
        .sum();
    let (baseline_test, test_metric) = match a.phase {
        Phase::Tune => {
            // A tuning run may validate test tensors, but must never forward a test sample.
            (None, None)
        }
        Phase::Full => {
            let mut frozen = Engine::load(&a.input, &a.plastic, cfg)?;
            configure_readout(&mut frozen, a.readout)?;
            let initial_test = evaluate(&mut frozen, &test, None)?;
            let mut pred = BufWriter::new(File::create(a.out.join("predictions.tsv"))?);
            write!(pred, "index\ttrue\tpredicted")?;
            for j in 0..a.actions {
                write!(pred, "\tp{j}")?;
            }
            writeln!(pred)?;
            let final_test = evaluate(&mut selected, &test, Some(&mut pred))?;
            pred.flush()?;
            (Some(initial_test), Some(final_test))
        }
    };
    let elapsed = start.elapsed().as_secs_f64();
    fn opt(v: Option<f64>) -> String {
        v.map(|x| format!("{x:.9}"))
            .unwrap_or_else(|| "null".into())
    }
    let json = format!(concat!(
        "{{\n  \"phase\": \"{}\",\n  \"mode\": \"{}\",\n  \"readout\": \"{}\",\n  \"seed\": {},\n",
        "  \"epochs_requested\": {},\n  \"best_epoch\": {},\n  \"train_count\": {},\n",
        "  \"validation_count\": {},\n  \"test_count\": {},\n  \"actions\": {},\n",
        "  \"learning_rate\": {:.9},\n  \"logit_gain\": {:.9},\n  \"active_fraction\": {:.9},\n  \"homeostasis\": {},\n  \"batch_size\": {},\n  \"patience\": {},\n",
        "  \"baseline_validation_accuracy\": {:.9},\n  \"baseline_validation_cross_entropy\": {:.9},\n",
        "  \"validation_accuracy\": {:.9},\n  \"validation_cross_entropy\": {:.9},\n  \"validation_correct_probability\": {:.9},\n",
        "  \"baseline_test_accuracy\": {},\n  \"baseline_test_cross_entropy\": {},\n",
        "  \"test_accuracy\": {},\n  \"test_cross_entropy\": {},\n  \"test_correct_probability\": {},\n",
        "  \"changed_plastic_weights\": {},\n  \"plastic_weight_l1_delta\": {:.9},\n  \"elapsed_seconds\": {:.3}\n}}\n"
    ), a.phase.name(), a.mode.name(), a.readout.name(), a.seed, a.epochs, best_epoch, train.len(), val.len(), test.len(), a.actions,
       a.lr, a.gain, a.active, a.homeostasis, a.batch_size, a.patience, baseline.acc, baseline.ce, best.acc, best.ce, best.prob,
       opt(baseline_test.map(|m| m.acc)), opt(baseline_test.map(|m| m.ce)),
       opt(test_metric.map(|m| m.acc)), opt(test_metric.map(|m| m.ce)), opt(test_metric.map(|m| m.prob)),
       changed, delta, elapsed);
    fs::write(a.out.join("metrics.json"), json)?;
    match test_metric {
        Some(m) => println!("best_epoch={best_epoch} val_accuracy={:.4} test_accuracy={:.4} changed_weights={changed} elapsed={elapsed:.1}s", best.acc, m.acc),
        None => println!("best_epoch={best_epoch} val_accuracy={:.4} changed_weights={changed} elapsed={elapsed:.1}s", best.acc),
    }
    Ok(())
}
fn main() {
    if let Err(e) = run() {
        eprintln!("train_classifier: {e}");
        std::process::exit(1)
    }
}
