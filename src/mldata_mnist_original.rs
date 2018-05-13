//! The "MNIST" database of handwritten digits from mldata.org.

use std::fs;
use std::path;

use app_dirs::*;
use ndarray::{Array1, Array2, Array3, ArrayView2, Axis, Zip};

use utils::downloader::assure_file;
use utils::error::Error;
use utils::hdf5;
use utils::hdf5::DynamicArray;

use canonical::CanonicalData;
use common::APP_INFO;

/// Configure the loader for the data set.
///
/// This structure implements the builder pattern to configure the [`DataSetLoader`].
pub struct DataSet {
    data_root: path::PathBuf,
    download: bool,
}

impl DataSet {
    pub fn new() -> Self {
        DataSet {
            data_root: get_app_dir(AppDataType::UserData, &APP_INFO, "mldata.org").unwrap(),
            download: true,
        }
    }

    pub fn create(&self) -> Result<DataSetLoader, Error> {
        DataSetLoader::new(&self.data_root, self.download)
    }

    pub fn data_root<P: AsRef<path::Path>>(&mut self, p: P) -> &mut Self {
        self.data_root = p.as_ref().into();
        self
    }

    pub fn download(&mut self, b: bool) -> &mut Self {
        self.download = b;
        self
    }
}

/// Load the data set.
///
/// The preferred way is to initialize this structure with [`DataSet`](struct.DataSet.html).
/// However, it is also possible to use [`new`](struct.DataSetLoader.html#method.new) and manually
/// set all options in the arguments.
pub struct DataSetLoader {
    data_file: path::PathBuf,
}

impl DataSetLoader {
    /// new
    pub fn new<P: AsRef<path::Path>>(data_path: P, download: bool) -> Result<DataSetLoader, Error> {
        let data_path = data_path.as_ref();
        fs::create_dir_all(data_path)?;

        let data_file = data_path.join("mnist-original.hdf5");

        if download {
            assure_file(&data_file, "http://mldata.org/repository/data/download/mnist-original")?;
        }

        Ok(DataSetLoader{
            data_file,
        })
    }

    pub fn load_data(&self) -> Result<Data, Error> {
        let file = hdf5::File::open(&self.data_file)?;

        let data = if let DynamicArray::UInt8(arr) = file.dataset("/data/data")?.read()? {
            arr
        } else {
            return Err(Error::DataType)
        };

        let label = if let DynamicArray::Float64(arr) = file.dataset("/data/label")?.read()? {
            arr
        } else {
            return Err(Error::DataType)
        };

        let x: Array3<u8> = data.t().into_shape((70000, 28, 28))?.to_owned();

        let y: Array1<_> = label.iter().map(|&f| f as u8).collect();

        Ok(Data::from(x, y))
    }
}

/// In-memory representation of the data
pub struct Data {
    x: Array3<u8>,
    y: Array1<u8>,
}

impl Data {
    fn from(x: Array3<u8>, y: Array1<u8>) -> Self {
        assert_eq!(x.len(), y.len() * 784);
        Data {
            x,
            y,
        }
    }

    pub fn n_samples(&self) -> usize {
        self.y.len()
    }

    pub fn get_sample(&self, idx: usize) -> (ArrayView2<u8>, u8) {
        let xv = self.x.subview(Axis(0), idx).reversed_axes();
        (xv, self.y[idx])
    }
}

impl CanonicalData for Data {
    fn to_canonical(&self) -> (Array2<f64>, Array2<f64>) {
        let mut x = Array3::zeros((self.y.len(), 28, 28));
        let mut y = Array2::zeros((self.y.len(), 1));

        Zip::from(&self.x)
            .and(&mut x)
            .apply(|xi, xf| *xf = *xi as f64);

        for (yi, yf) in self.y.iter().zip(y.iter_mut()) {
            *yf = *yi as f64
        }

        let mut tmp = Array2::zeros((28, 28));

        for mut img in x.outer_iter_mut() {
            tmp.assign(&img.t());
            img.assign(&tmp);
        }
        (x.into_shape((self.y.len(), 784)).unwrap(), y)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Array2;
    use super::*;

    #[test]
    fn load() {
        let data = DataSet::new().download(true).create().unwrap();
        let tst = data.load_data().unwrap();
        assert_eq!(tst.n_samples(), 70000);

        assert_eq!(tst.get_sample(4150).1, 0);

        let x = X_4150.to_vec();
        assert_eq!(tst.get_sample(4150).0, Array2::from_shape_vec((28, 28), x).unwrap());

        assert_eq!(tst.get_sample(30000).1, 4);
    }

    #[test]
    fn canonical() {
        let data = DataSet::new().download(true).create().unwrap();

        let (x, y) = data.load_data().unwrap().into_canonical();
        assert_eq!(x.shape(), [70000, 28 * 28]);
        assert_eq!(y.shape(), [70000, 1]);

        let xv = X_4150.iter().map(|&u| u as f64).collect();
        assert_eq!(x.subview(Axis(0), 4150), Array1::from_shape_vec(784, xv).unwrap());

        assert_eq!(y[(30000, 0)], 4.0);
    }

    const X_4150: [u8; 784] = [
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 255, 253, 253, 133,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 160, 208, 228, 206, 252, 231,  59,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   8,  68, 231, 252, 68,   72, 252, 252, 225, 164, 164,  23,   4,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,  15, 189, 252, 252, 252, 253, 252, 252, 252, 252, 252, 252, 252, 128,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,  23, 183, 252, 252, 252, 252, 253, 252, 247, 177, 177, 221, 217, 252, 220,  31,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,  93, 252, 252, 252, 252, 234,  74,  74,  69,   0,   0,  44, 134, 252, 252, 103,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,  64, 238, 252, 252, 252, 252, 103,   0,   0,   0,   0,   0,   0, 134, 252, 252, 103,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   6, 166, 252, 252, 204,  22,  14,   6,   0,   0,   0,   0,   0,   0, 134, 252, 236,  67,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,  42, 252, 252, 246,  97,   0,   0,   0,   0,   0,   0,   0,   0,  23, 224, 252, 207,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,  36, 216, 252, 252, 133,   0,   0,   0,   0,   0,   0,   0,   0,   0, 119, 252, 252, 207,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0, 122, 253, 253, 178,   0,   0,   0,   0,   0,   0,   0,   0,   0,  31, 210, 253, 216,  35,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0, 208, 252, 252, 142,   0,   0,   0,   0,   0,   0,   0,   0,  29, 204, 252, 252,  40,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,  68, 236, 252, 244,  27,   0,   0,   0,   0,   0,   0,   4,  24, 205, 252, 232,  88,   5,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0, 105, 252, 244,  70,   0,   0,   0,   0,   0,   0,   0, 130, 252, 252, 252, 151,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0, 105, 252, 237,   0,   0,   0,   0,   0,   0,  45, 224, 247, 252, 252, 185,  52,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0, 210, 252, 237,   0,   0,   0,   0,  29, 144, 222, 253, 252, 250, 151,   9,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0, 253, 252, 237,   0,   0,  16, 134, 245, 252, 252, 253, 220, 111,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0, 113, 252, 251, 238, 238, 240, 252, 252, 252, 252, 163,  36,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,  25, 192, 252, 252, 252, 252, 252, 252, 217, 120,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,  25, 199, 252, 252, 252, 252, 111,  24,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    ];
}
