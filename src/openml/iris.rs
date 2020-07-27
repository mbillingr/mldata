//! The "Iris" data set.

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

        let data_file = data_path.join("dataset_61_iris.arff");

        if download {
            assure_file(&data_file, "https://www.openml.org/data/download/61/dataset_61_iris.arff")?;
        }

        Ok(DataSetLoader{
            data_file,
        })
    }

    pub fn load_data(&self) -> Result<IrisData, Error> {
        let mut file =fs::File::open(&self.data_file)?;
        let mut input = String::new();
        file.read_to_string(&mut input)?;

        let data = arff::from_str(&input)?;
        Ok(data)
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Deserialize)]
pub enum Iris {
    #[serde(rename = "Iris-setosa")]
    Setosa,

    #[serde(rename = "Iris-versicolor")]
    Versicolor,

    #[serde(rename = "Iris-virginica")]
    Virginica,
}

/// A single specimen of the Iris data set
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct IrisRow {
    #[serde(rename = "sepallength")]
    pub sepal_length: f32,

    #[serde(rename = "sepalwidth")]
    pub sepal_width: f32,

    #[serde(rename = "petallength")]
    pub petal_length: f32,

    #[serde(rename = "petalwidth")]
    pub petal_width: f32,

    pub class: Iris,
}

/// The Iris data set
pub type IrisData = Vec<IrisRow>;

impl CanonicalData for IrisData {
    fn to_canonical(&self) -> (Array2<f64>, Array2<f64>) {
        let mut x = Array2::zeros([self.len(), 4]);
        let mut y = Array2::zeros((self.len(), 1));

        Zip::from(x.genrows_mut())
            .and(y.genrows_mut())
            .and(&self[..])
            .apply(|mut x, mut y, r| {
                x[0] = r.sepal_length as f64;
                x[1] = r.sepal_width as f64;
                x[2] = r.petal_length as f64;
                x[3] = r.petal_width as f64;
                y[0] = (r.class as usize + 1) as f64;
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
        let iris = data.load_data().unwrap();
        assert_eq!(iris.len(), 150);
        // check class labels of a few specific samples
        assert_eq!(iris[25].class, Iris::Setosa);
        assert_eq!(iris[75].class, Iris::Versicolor);
        assert_eq!(iris[125].class, Iris::Virginica);
        // check one arbirtrary example
        assert_eq!(iris[140], IrisRow {
            sepal_length: 6.7,
            sepal_width: 3.1,
            petal_length: 5.6,
            petal_width: 2.4,
            class: Iris::Virginica,
        });
    }

    #[test]
    fn canonical() {
        let data = DataSet::new().download(true).create().unwrap();

        let (x, y) = data.load_data().unwrap().into_canonical();
        assert_eq!(x.shape(), [150, 4]);
        assert_eq!(y.shape(), [150, 1]);

        // check class labels of a few specific samples
        assert_eq!(y[[25, 0]], 1.0);
        assert_eq!(y[[75, 0]], 2.0);
        assert_eq!(y[[125, 0]], 3.0);
    }
}
