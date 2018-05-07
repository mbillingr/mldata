//! The "Auto MPG" data set from mldata.org.

use std::fs;
use std::path;

use app_dirs::*;
use ndarray::{Array2, Zip};

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

        let data_file = data_path.join("uci-20070111-autompg.hdf5");

        if download {
            assure_file(&data_file, "http://mldata.org/repository/data/download/uci-20070111-autompg")?;
        }

        Ok(DataSetLoader{
            data_file,
        })
    }

    pub fn load_data(&self) -> Result<Data, Error> {
        let file = hdf5::File::open(&self.data_file)?;

        let int0 = if let DynamicArray::Int32(arr) = file.dataset("/data/int0")?.read()? {
            arr
        } else {
            return Err(Error::Internal)
        };

        let double1 = if let DynamicArray::Float64(arr) = file.dataset("/data/double1")?.read()? {
            arr
        } else {
            return Err(Error::Internal)
        };

        let int2 = if let DynamicArray::Int32(arr) = file.dataset("/data/int2")?.read()? {
            arr
        } else {
            return Err(Error::Internal)
        };

        let mut x = Vec::new();
        let mut y = Vec::new();

        Zip::from(int0.gencolumns())
            .and(&double1)
            .and(int2.gencolumns())
            .apply(|i0, d1, i2| {
                let yi = TargetVar {
                    mpg: i2[2],
                };

                let xi = FeatureRow {
                    cylinders: i0[0],
                    displacement: i0[1],
                    horsepower: match i0[2] {
                        -2147483648 => ::std::f64::NAN,
                        nr => nr as f64,
                    },
                    weight: i0[3],
                    acceleration: *d1,
                    model_year: i2[0],
                    origin: i2[1],
                };

                x.push(xi);
                y.push(yi);
            });

        Ok(Data::from(x, y))
    }
}

#[derive(Debug, PartialEq)]
pub struct FeatureRow {
    pub cylinders: i32,
    pub displacement: i32,
    pub horsepower: f64,
    pub weight: i32,
    pub acceleration: f64,
    pub model_year: i32,
    pub origin: i32,
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct TargetVar {
    pub mpg: i32,
}

/// In-memory representation of the data
pub struct Data {
    x: Vec<FeatureRow>,
    y: Vec<TargetVar>,
    n_samples: usize,
}

impl Data {
    fn from(x: Vec<FeatureRow>, y: Vec<TargetVar>) -> Self {
        assert_eq!(x.len(), y.len());
        Data {
            n_samples: y.len(),
            x,
            y,
        }
    }

    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    pub fn get_sample(&self, idx: usize) -> (&FeatureRow, TargetVar) {
        (&self.x[idx], self.y[idx])
    }
}

impl CanonicalData for Data {
    fn to_canonical(&self) -> (Array2<f64>, Array2<f64>) {
        let mut x_tmp = Vec::new();
        for xi in self.x.iter() {
            x_tmp.push(xi.cylinders as f64);
            x_tmp.push(xi.displacement as f64);
            x_tmp.push(xi.horsepower as f64);
            x_tmp.push(xi.weight as f64);
            x_tmp.push(xi.acceleration as f64);
            x_tmp.push(xi.model_year as f64);
            x_tmp.push(xi.origin as f64);
        }
        let y_tmp = self.y.iter().map(|yi| yi.mpg as f64).collect();
        let x = Array2::from_shape_vec((self.n_samples, 7), x_tmp).unwrap();
        let y = Array2::from_shape_vec((self.n_samples, 1), y_tmp).unwrap();
        (x, y)
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load() {
        let data = DataSet::new().download(true).create().unwrap();
        let tst = data.load_data().unwrap();
        assert_eq!(tst.n_samples, 398);
        assert_eq!(tst.get_sample(41), (
            &FeatureRow {
                cylinders: 8,
                displacement: 318,
                horsepower: 150.0,
                weight: 4096,
                acceleration: 13.0,
                model_year: 71,
                origin: 1,
            },
            TargetVar {
                mpg: 14
            }
        ));
    }

    #[test]
    fn canonical() {
        let data = DataSet::new().download(true).create().unwrap();

        let (x, y) = data.load_data().unwrap().into_canonical();
        assert_eq!(x.shape(), [398, 7]);
        assert_eq!(y.shape(), [398, 1]);

        assert_eq!(y[[41, 0]], 14.0);
        assert_eq!(x[[41, 0]], 8.0);
        assert_eq!(x[[41, 1]], 318.0);
        assert_eq!(x[[41, 2]], 150.0);
        assert_eq!(x[[41, 3]], 4096.0);
        assert_eq!(x[[41, 4]], 13.0);
        assert_eq!(x[[41, 5]], 71.0);
        assert_eq!(x[[41, 6]], 1.0);
    }
}
