//! The "MNIST" data set.

use std::fs;
use std::io::Read;
use std::path;

use app_dirs::*;
use arff;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayView3, Axis, ShapeBuilder};

use utils::downloader::assure_file;
use utils::error::Error;

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
            data_root: get_app_dir(AppDataType::UserData, &APP_INFO, "openml.org").unwrap(),
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

        let data_file = data_path.join("mnist_784.arff");

        if download {
            assure_file(&data_file, "https://www.openml.org/data/download/52667/mnist_784.arff")?;
        }

        Ok(DataSetLoader{
            data_file,
        })
    }

    pub fn load_data(&self) -> Result<MNISTData, Error> {
        let mut file =fs::File::open(&self.data_file)?;
        let mut input = String::new();
        file.read_to_string(&mut input)?;

        let raw_data: Vec<u8> = arff::flat_from_str(&input)?;

        /*let x = ArrayView2::from_shape([70000, 784].strides([785, 1]), &raw_data[..])?;
        let x2d = ArrayView3::from_shape([70000, 28, 28].strides([785, 28, 1]), raw_data.as_ref())?;
        let y = ArrayView1::from_shape([70000].strides([785]), raw_data.as_ref())?;*/

        Ok(MNISTData{
            raw_data,
        })
    }
}

/// A single image in the MNIST data set
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct MNISTRow([[u8; 28]; 28], u8);

/// The MNIST data set
pub struct MNISTData {
    raw_data: Vec<u8>,
}

impl MNISTData {
    pub fn x(&self) -> ArrayView2<u8> {
        ArrayView2::from_shape([70000, 784].strides([785, 1]), &self.raw_data[..]).unwrap()
    }

    pub fn x2d(&self) -> ArrayView3<u8> {
        ArrayView3::from_shape([70000, 28, 28].strides([785, 28, 1]), &self.raw_data[..]).unwrap()
    }

    pub fn y(&self) -> ArrayView1<u8> {
        ArrayView1::from_shape([70000].strides([785]), &self.raw_data[784..]).unwrap()
    }
}

impl CanonicalData for MNISTData {
    fn to_canonical(&self) -> (Array2<f64>, Array2<f64>) {
        let mut x = Array2::zeros([70000, 784]);
        let mut y = Array2::zeros((70000, 1));

        for (xo, xi) in x.iter_mut().zip(self.x().iter()) {
            *xo = *xi as f64
        }

        for (yo, yi) in y.iter_mut().zip(self.y().iter()) {
            *yo = *yi as f64
        }

        (x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load() {
        let data = DataSet::new().download(true).create().unwrap();
        let mnist = data.load_data().unwrap();

        let x = mnist.x();
        let x2d = mnist.x2d();
        let y = mnist.y();

        assert_eq!(x.raw_dim(), [70000, 784]);
        assert_eq!(x2d.raw_dim(), [70000, 28, 28]);
        assert_eq!(y.raw_dim(), [70000]);

        let x_ref =  ArrayView1::from_shape(784, &X_42).unwrap();
        let x2_ref =  ArrayView2::from_shape((28, 28), &X_42).unwrap();

        assert_eq!(x.subview(Axis(0), 42), x_ref);
        assert_eq!(x2d.subview(Axis(0), 42), x2_ref);
        assert_eq!(y[42], 7);
    }

    #[test]
    fn canonical() {
        let data = DataSet::new().download(true).create().unwrap();

        let (x, y) = data.load_data().unwrap().into_canonical();
        assert_eq!(x.shape(), [70000, 28 * 28]);
        assert_eq!(y.shape(), [70000, 1]);

        let xv = X_42.iter().map(|&u| u as f64).collect();
        assert_eq!(x.subview(Axis(0), 42), Array1::from_shape_vec(784, xv).unwrap());
        assert_eq!(y[(42, 0)], 7.0);
    }

    const X_42: [u8; 784] = [
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1, 26,111,195,230, 30,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 28,107,195,254,254,254,244, 20,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0, 46,167,248,254,222,146,150,254,174,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0, 65,223,246,254,153, 61, 10,  0, 48,254,129,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0, 85,175,164, 80,  2,  0,  0,  0, 48,254,120,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,182,254, 16,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,207,254, 16,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,207,202,  3,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 28,248,170,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,107,254, 61,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,166,252, 30,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,191,206,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,191,206,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 14,246,186,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 91,254, 77,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,175,254, 48,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,175,240, 27,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,215,222,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,115,255,152,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,134,255, 68,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
        0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0];
}
