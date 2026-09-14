//! Python facade for the topology-defined engine. No Python callback enters a tick.
use novi_core::plastic::{Engine as CoreEngine, PlasticConfig};
use pyo3::exceptions::{PyOSError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::{Path, PathBuf};

fn valid_rates(rates: &[f32], dimension: usize) -> PyResult<()> {
    if rates.len() != dimension || rates.iter().any(|&x| !x.is_finite() || x < 0.0) {
        Err(PyValueError::new_err(
            "rates must be finite nonnegative values in checkpoint input-port order",
        ))
    } else {
        Ok(())
    }
}

#[pyclass(module = "novigrad._native", name = "Engine")]
struct PyEngine {
    inner: CoreEngine,
}

#[pymethods]
impl PyEngine {
    #[staticmethod]
    fn load(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        if !path.exists() {
            return Err(PyOSError::new_err(format!(
                "checkpoint not found: {}",
                path.display()
            )));
        }
        py.detach(|| CoreEngine::load_checkpoint(&path))
            .map(|inner| Self { inner })
            .map_err(PyRuntimeError::new_err)
    }

    #[staticmethod]
    #[pyo3(signature = (input_edges, plastic_edges, *, actions=2, learning_rate=0.02, logit_gain=6.0, active_fraction=0.1, weight_limit=1.0, homeostasis=true, readout="positive"))]
    #[allow(clippy::too_many_arguments)]
    fn from_edges(
        py: Python<'_>,
        input_edges: PathBuf,
        plastic_edges: PathBuf,
        actions: usize,
        learning_rate: f32,
        logit_gain: f32,
        active_fraction: f32,
        weight_limit: f32,
        homeostasis: bool,
        readout: &str,
    ) -> PyResult<Self> {
        if !input_edges.exists() || !plastic_edges.exists() {
            return Err(PyOSError::new_err("input or plastic edge file not found"));
        }
        if !matches!(readout, "positive" | "opponent") {
            return Err(PyValueError::new_err(
                "readout must be positive or opponent",
            ));
        }
        let config = PlasticConfig {
            actions,
            learning_rate,
            logit_gain,
            active_fraction,
            weight_limit,
            homeostasis,
        };
        let mut inner = py
            .detach(|| CoreEngine::load(&input_edges, &plastic_edges, config))
            .map_err(PyValueError::new_err)?;
        if readout == "opponent" {
            let gains: Vec<f32> = (0..inner.output_ids().len())
                .map(|i| {
                    if (i / actions).is_multiple_of(2) {
                        1.0
                    } else {
                        -1.0
                    }
                })
                .collect();
            inner
                .set_output_gains(&gains)
                .map_err(PyValueError::new_err)?;
        }
        Ok(Self { inner })
    }

    #[getter]
    fn input_ids(&self) -> Vec<u64> {
        self.inner.input_ids().to_vec()
    }
    #[getter]
    fn hidden_ids(&self) -> Vec<u64> {
        self.inner.hidden_ids().to_vec()
    }
    #[getter]
    fn output_ids(&self) -> Vec<u64> {
        self.inner.output_ids().to_vec()
    }
    #[getter]
    fn output_actions(&self) -> Vec<usize> {
        self.inner.output_actions().to_vec()
    }
    #[getter]
    fn output_gains(&self) -> Vec<f32> {
        self.inner.output_gains().to_vec()
    }
    #[getter]
    fn weights(&self) -> Vec<f32> {
        self.inner.weights().collect()
    }
    #[getter]
    fn config<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let c = self.inner.config();
        let d = PyDict::new(py);
        d.set_item("actions", c.actions)?;
        d.set_item("learning_rate", c.learning_rate)?;
        d.set_item("logit_gain", c.logit_gain)?;
        d.set_item("active_fraction", c.active_fraction)?;
        d.set_item("weight_limit", c.weight_limit)?;
        d.set_item("homeostasis", c.homeostasis)?;
        Ok(d)
    }

    fn set_output_actions(&mut self, actions: Vec<usize>) -> PyResult<()> {
        self.inner
            .set_output_actions(&actions)
            .map_err(PyValueError::new_err)
    }
    fn set_output_gains(&mut self, gains: Vec<f32>) -> PyResult<()> {
        self.inner
            .set_output_gains(&gains)
            .map_err(PyValueError::new_err)
    }

    fn infer(&mut self, py: Python<'_>, rates: Vec<f32>) -> PyResult<Vec<f32>> {
        valid_rates(&rates, self.inner.input_ids().len())?;
        Ok(py.detach(|| {
            let probabilities = self.inner.forward(&rates).to_vec();
            self.inner.clear_pending();
            probabilities
        }))
    }

    fn infer_batch(&mut self, py: Python<'_>, rows: Vec<Vec<f32>>) -> PyResult<Vec<Vec<f32>>> {
        if rows.is_empty() {
            return Err(PyValueError::new_err("batch must be nonempty"));
        }
        for row in &rows {
            valid_rates(row, self.inner.input_ids().len())?;
        }
        Ok(py.detach(|| {
            rows.iter()
                .map(|row| {
                    let p = self.inner.forward(row).to_vec();
                    self.inner.clear_pending();
                    p
                })
                .collect()
        }))
    }

    /// Apply one mean reward gradient at the current batch-start weights.
    fn learn(
        &mut self,
        py: Python<'_>,
        rows: Vec<Vec<f32>>,
        actions: Vec<usize>,
        rewards: Vec<f32>,
    ) -> PyResult<()> {
        if rows.is_empty() || rows.len() != actions.len() || rows.len() != rewards.len() {
            return Err(PyValueError::new_err(
                "rows, actions, rewards must have equal nonzero lengths",
            ));
        }
        for ((row, &action), &reward) in rows.iter().zip(&actions).zip(&rewards) {
            valid_rates(row, self.inner.input_ids().len())?;
            if action >= self.inner.config().actions || !reward.is_finite() {
                return Err(PyValueError::new_err(
                    "action out of range or nonfinite reward",
                ));
            }
        }
        py.detach(|| {
            for ((row, &action), &reward) in rows.iter().zip(&actions).zip(&rewards) {
                self.inner.forward(row);
                if let Err(error) = self.inner.accumulate_reward(action, reward) {
                    self.inner.discard_batch();
                    return Err(error);
                }
            }
            if let Err(error) = self.inner.apply_batch() {
                self.inner.discard_batch();
                return Err(error);
            }
            Ok(())
        })
        .map_err(PyRuntimeError::new_err)
    }

    #[pyo3(signature = (path, *, overwrite=false))]
    fn save(&self, py: Python<'_>, path: PathBuf, overwrite: bool) -> PyResult<()> {
        let path = path.as_path();
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        py.detach(|| {
            let temporary = tempfile::NamedTempFile::new_in(parent).map_err(|e| e.to_string())?;
            self.inner.save_checkpoint(temporary.path())?;
            std::fs::File::open(temporary.path())
                .and_then(|file| file.sync_all())
                .map_err(|e| e.to_string())?;
            if overwrite {
                temporary.persist(path).map_err(|e| e.to_string())?;
            } else {
                temporary
                    .persist_noclobber(path)
                    .map_err(|e| e.to_string())?;
            }
            Ok::<(), String>(())
        })
        .map_err(PyOSError::new_err)
    }
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEngine>()?;
    Ok(())
}
