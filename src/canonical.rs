//! Home module of the canonical data representation

use ndarray::Array2;

/// Conversion into canonical data representation.
///
/// The canonical representation of a data set are two 2D arrays X and Y. They contain the features
/// and target variables, respectively. Each row in X and Y correspond to the same sample. Some data
/// sets do not have target variables, in this case Y can be empty.
///
/// This representation was chosen to be generic. It should be possible to represent most data sets
/// using `f64` and missing data can be encoded as NaN.
pub trait CanonicalData {
    fn to_canonical(&self) -> (Array2<f64>, Array2<f64>);

    fn into_canonical(self) -> (Array2<f64>, Array2<f64>)
        where Self: Sized
    {
        self.to_canonical()
    }
}
