//! The "Boston Housing" data set from mldata.org (regression-datasets housing).

use std::fs;
use std::path;

use app_dirs::*;
use ndarray::{Array2, Zip};

use utils::downloader::assure_file;
use utils::error::Error;
use utils::hdf5;

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

        let data_file = data_path.join("regression-datasets-housing.hdf5");

        if download {
            assure_file(&data_file, "http://mldata.org/repository/data/download/regression-datasets-housing")?;
        }

        Ok(DataSetLoader{
            data_file,
        })
    }

    pub fn load_data(&self) -> Result<Data, Error> {
        let file = hdf5::File::open(&self.data_file)?;

        let double0 = file.dataset("/data/double0")?.read_f64()?;
        let int1 = file.dataset("/data/int1")?.read_i32()?;
        let double2 = file.dataset("/data/double2")?.read_f64()?;
        let int3 = file.dataset("/data/int3")?.read_i32()?;
        let double4 = file.dataset("/data/double4")?.read_f64()?;
        let int5 = file.dataset("/data/int5")?.read_i32()?;
        let double6 = file.dataset("/data/double6")?.read_f64()?;

        let mut x = Vec::new();
        let mut y = Vec::new();

        Zip::from(&double0)
            .and(&int1)
            .and(&double2)
            //.and(&int3)
            .and(double4.gencolumns())
            .and(int5.gencolumns())
            .and(double6.gencolumns())
            .apply(|d0, i1, d2, d4, i5, d6| {
                let yi = TargetVar {
                    medv: d6[2],
                };

                let xi = FeatureRow {
                    crim: *d0,
                    zn: *i1,
                    indus: *d2,
                    chas: false,  // placeholder
                    nox: d4[0],
                    rm: d4[1],
                    age: d4[2],
                    dis: d4[3],
                    rad: i5[0],
                    tax: i5[1],
                    ptratio: i5[2],
                    b: d6[0],
                    lstat: d6[1],
                };

                x.push(xi);
                y.push(yi);
            });

        for (xi, c) in x.iter_mut().zip(&int3) {
            xi.chas = *c > 0;
        }

        Ok(Data::from(x, y))
    }
}

#[derive(Debug, PartialEq)]
pub struct FeatureRow {
    /// per capita crime rate by town
    pub crim: f64,

    /// proportion of residential land zoned for lots over 25,000 sq.ft.
    pub zn: i32,

    /// proportion of non-retail business acres per town
    pub indus: f64,

    /// Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
    pub chas: bool,

    /// nitric oxides concentration (parts per 10 million)
    pub nox: f64,

    /// average number of rooms per dwelling
    pub rm: f64,

    /// proportion of owner-occupied units built prior to 1940
    pub age: f64,

    /// weighted distances to five Boston employment centres
    pub dis: f64,

    /// index of accessibility to radial highways
    pub rad: i32,

    /// full-value property-tax rate per $10,000
    pub tax: i32,

    /// pupil-teacher ratio by town
    pub ptratio: i32,

    /// 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
    pub b: f64,

    /// % lower status of the population
    pub lstat: f64,
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct TargetVar {
    /// Median value of owner-occupied homes in $1000's
    pub medv: f64,
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
            x_tmp.push(xi.crim as f64);
            x_tmp.push(xi.zn as f64);
            x_tmp.push(xi.indus as f64);
            x_tmp.push(if xi.chas {1.0} else {0.0});
            x_tmp.push(xi.nox as f64);
            x_tmp.push(xi.rm as f64);
            x_tmp.push(xi.age as f64);
            x_tmp.push(xi.dis as f64);
            x_tmp.push(xi.rad as f64);
            x_tmp.push(xi.tax as f64);
            x_tmp.push(xi.ptratio as f64);
            x_tmp.push(xi.b as f64);
            x_tmp.push(xi.lstat as f64);
        }
        let y_tmp = self.y.iter().map(|yi| yi.medv).collect();
        let x = Array2::from_shape_vec((self.n_samples, 13), x_tmp).unwrap();
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
        assert_eq!(tst.n_samples, 506);
        assert_eq!(tst.get_sample(6), (
            &FeatureRow {
                crim: 0.08829,
                zn: 12,
                indus: 7.87,
                chas: false,
                nox: 0.524,
                rm: 6.012,
                age: 66.6,
                dis: 5.5605,
                rad: 5,
                tax: 311,
                ptratio: 15,
                b: 395.6,
                lstat: 12.43,
            },
            TargetVar {
                medv: 22.9
            }
        ));
    }

    #[test]
    fn canonical() {
        let data = DataSet::new().download(true).create().unwrap();

        let (x, y) = data.load_data().unwrap().into_canonical();
        assert_eq!(x.shape(), [506, 13]);
        assert_eq!(y.shape(), [506, 1]);

        assert_eq!(y[[6, 0]], 22.9);
        assert_eq!(x[[6, 0]], 0.08829);
        assert_eq!(x[[6, 1]], 12.0);
        assert_eq!(x[[6, 2]], 7.87);
        assert_eq!(x[[6, 3]], 0.0);
        assert_eq!(x[[6, 4]], 0.524);
        assert_eq!(x[[6, 5]], 6.012);
        assert_eq!(x[[6, 6]], 66.6);
        assert_eq!(x[[6, 7]], 5.5605);
        assert_eq!(x[[6, 8]], 5.0);
        assert_eq!(x[[6, 9]], 311.0);
        assert_eq!(x[[6, 10]], 15.0);
        assert_eq!(x[[6, 11]], 395.6);
        assert_eq!(x[[6, 12]], 12.43);
    }
}
