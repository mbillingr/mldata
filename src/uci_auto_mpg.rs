//! The "Auto MPG" data set.

use std::fs;
use std::io::{BufRead, BufReader, Read};
use std::path;

use app_dirs::*;
use ndarray::Array2;

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
            data_root: get_app_dir(AppDataType::UserData, &APP_INFO, "UCI/auto_mpg").unwrap(),
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
    info_file: path::PathBuf,
}

impl DataSetLoader {
    /// new
    pub fn new<P: AsRef<path::Path>>(data_path: P, download: bool) -> Result<DataSetLoader, Error> {
        let data_path = data_path.as_ref();
        fs::create_dir_all(data_path)?;

        let data_file = data_path.join("auto_mpg.data");
        let info_file = data_path.join("auto_mpg.names");

        if download {
            assure_file(&data_file, "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")?;
            assure_file(&info_file, "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.names")?;
        }

        Ok(DataSetLoader{
            data_file,
            info_file,
        })
    }

    pub fn load_info(&self) -> Result<String, Error> {
        let mut file = fs::File::open(&self.info_file)?;

        let mut info = String::new();
        file.read_to_string(&mut info)?;

        Ok(info)
    }

    pub fn load_data(&self) -> Result<Data, Error> {
        let input = BufReader::new(fs::File::open(&self.data_file)?);

        let mut x = Vec::new();
        let mut y = Vec::new();

        for line in input.lines() {
            let line = line?;
            if line.is_empty() {
                continue
            }

            let mut cols = line.split_whitespace();

            let yi = TargetVar {
                mpg: cols.next().unwrap().parse().unwrap(),
            };

            let xi = FeatureRow {
                cylinders: cols.next().unwrap().parse().unwrap(),
                displacement: cols.next().unwrap().parse().unwrap(),
                horsepower: match cols.next().unwrap() {
                    "?" => ::std::f32::NAN,
                    nr => nr.parse().unwrap(),
                },
                weight: cols.next().unwrap().parse().unwrap(),
                acceleration: cols.next().unwrap().parse().unwrap(),
                model_year: cols.next().unwrap().parse().unwrap(),
                origin: cols.next().unwrap().parse().unwrap(),
                car_name: {
                    let name = cols.collect::<Vec<_>>().join(" ");
                    name[1..name.len()-1].to_owned()  // remove enclosing "s
                }
            };

            x.push(xi);
            y.push(yi);
        }

        Ok(Data::from(x, y))
    }
}

#[derive(Debug, PartialEq)]
pub struct FeatureRow {
    pub cylinders: u8,
    pub displacement: f32,
    pub horsepower: f32,
    pub weight: f32,
    pub acceleration: f32,
    pub model_year: u8,
    pub origin: u8,
    pub car_name: String,
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct TargetVar {
    pub mpg: f32,
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
        let data = DataSet::new().download(false).create().unwrap();
        let tst = data.load_data().unwrap();
        assert_eq!(tst.n_samples, 398);
        assert_eq!(tst.get_sample(41), (
            &FeatureRow {
                cylinders: 8,
                displacement: 318.0,
                horsepower: 150.0,
                weight: 4096.0,
                acceleration: 13.0,
                model_year: 71,
                origin: 1,
                car_name: String::from("plymouth fury iii"),
            },
            TargetVar {
                mpg: 14.0
            }
        ));
    }

    #[test]
    fn canonical() {
        let data = DataSet::new().download(false).create().unwrap();

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
