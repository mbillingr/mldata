//! The "Iris" data set.

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
            data_root: get_app_dir(AppDataType::UserData, &APP_INFO, "UCI/iris").unwrap(),
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

        let data_file = data_path.join("iris.data");
        let info_file = data_path.join("iris.names");

        if download {
            assure_file(&data_file, "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")?;
            assure_file(&info_file, "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.names")?;
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
            let elements: Vec<_> = line.split(",").collect();
            for e in &elements[0..4] {
                x.push(e.parse().unwrap());
            }
            y.push(elements[4].into());
        }

        Ok(Data::from(x, y))
    }
}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub enum Iris {
    Setosa,
    Versicolor,
    Virginica,
}

impl<'a> From<&'a str> for Iris {
    fn from(s: &str) -> Iris {
        match s {
            "Iris-setosa" => Iris::Setosa,
            "Iris-versicolor" => Iris::Versicolor,
            "Iris-virginica" => Iris::Virginica,
            _ => panic!("Cannot convert string to Iris")
        }
    }
}

/// In-memory representation of the data
pub struct Data {
    x: Vec<f32>,
    y: Vec<Iris>,
    n_samples: usize,
}

impl Data {
    fn from(x: Vec<f32>, y: Vec<Iris>) -> Self {
        assert_eq!(x.len(), y.len() * 4);
        Data {
            n_samples: y.len(),
            x,
            y,
        }
    }

    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    pub fn get_sample(&self, idx: usize) -> (&[f32], Iris) {
        assert!(idx < self.n_samples);
        let start = 4 * idx;
        (&self.x[start..start+4], self.y[idx])
    }
}

impl CanonicalData for Data {
    fn to_canonical(&self) -> (Array2<f64>, Array2<f64>) {
        let x_tmp = self.x.iter().map(|f| *f as f64).collect();
        let y_tmp = self.y.iter().map(|f| *f as usize as f64).collect();
        let x = Array2::from_shape_vec((self.n_samples, 4), x_tmp).unwrap();
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
        assert_eq!(tst.n_samples, 150);
        // check class labels of a few specific samples
        assert_eq!(tst.get_sample(25).1, Iris::Setosa);
        assert_eq!(tst.get_sample(75).1, Iris::Versicolor);
        assert_eq!(tst.get_sample(125).1, Iris::Virginica);
    }

    #[test]
    fn canonical() {
        let data = DataSet::new().download(true).create().unwrap();

        let (x, y) = data.load_data().unwrap().into_canonical();
        assert_eq!(x.shape(), [150, 4]);
        assert_eq!(y.shape(), [150, 1]);

        // check class labels of a few specific samples
        assert_eq!(y[[25, 0]], 0.0);
        assert_eq!(y[[75, 0]], 1.0);
        assert_eq!(y[[125, 0]], 2.0);
    }
}
