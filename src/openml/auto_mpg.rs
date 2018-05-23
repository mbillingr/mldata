//! The "Auto MPG" data set.

use std::f64;
use std::fs;
use std::io::Read;
use std::path;

use app_dirs::*;
use arff;
use ndarray::{Array2, Zip};

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

        let data_file = data_path.join("dataset_2182_autoMpg.arff");

        if download {
            assure_file(&data_file, "https://www.openml.org/data/download/3633/dataset_2182_autoMpg.arff")?;
        }

        Ok(DataSetLoader{
            data_file,
        })
    }

    pub fn load_data(&self) -> Result<AutoMpgData, Error> {
        let mut file =fs::File::open(&self.data_file)?;
        let mut input = String::new();
        file.read_to_string(&mut input)?;

        let data = arff::from_str(&input)?;
        Ok(data)
    }
}

/// A single specimen of the Iris data set
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct AutoMpgRow {
    cylinders: u8,
    displacement: f32,
    horsepower: Option<f32>,  // horsepower has 6 missing values
    weight: f32,
    acceleration: f32,
    model: u8,
    origin: u8,

    #[serde(rename = "class")]
    mpg: f32
}

/// The Iris data set
pub type AutoMpgData = Vec<AutoMpgRow>;

impl CanonicalData for AutoMpgData {
    fn to_canonical(&self) -> (Array2<f64>, Array2<f64>) {
        let mut x = Array2::zeros([self.len(), 7]);
        let mut y = Array2::zeros((self.len(), 1));

        Zip::from(x.genrows_mut())
            .and(y.genrows_mut())
            .and(&self[..])
            .apply(|mut x, mut y, r| {
                x[0] = r.cylinders as f64;
                x[1] = r.displacement as f64;
                x[2] = r.horsepower.map(|ps| ps as f64).unwrap_or(f64::NAN);
                x[3] = r.weight as f64;
                x[4] = r.acceleration as f64;
                x[5] = r.model as f64;
                x[6] = r.origin as f64;
                y[0] = r.mpg as f64;
            });

        (x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load() {
        let data = DataSet::new().download(true).create().unwrap();
        let autos = data.load_data().unwrap();
        assert_eq!(autos.len(), 398);

        assert_eq!(
            autos[41],
            AutoMpgRow {
                cylinders: 8,
                displacement: 318.0,
                horsepower: Some(150.0),
                weight: 4096.0,
                acceleration: 13.0,
                model: 71,
                origin: 1,
                mpg: 14.0
            }
        );
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
