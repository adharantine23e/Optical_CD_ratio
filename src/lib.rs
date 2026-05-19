mod utils;
pub mod feature_extract;
use pyo3::prelude::*;


/// A Python module implemented in Rust.
#[pymodule]
mod cup_disk_ratio_api {
    use pyo3::prelude::*;

    /// Formats the sum of two numbers as string.
    #[pyfunction]
    fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
        Ok((a + b).to_string())
    }
}
