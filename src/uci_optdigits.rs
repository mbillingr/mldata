//! The "Optical Recognition of Handwritten Digits" data set.

use std::fs;
use std::io::Read;
use std::path;

use app_dirs::*;
use ndarray::{Array2, ArrayView2, ShapeBuilder, Zip};

use utils::downloader::assure_file;
use utils::error::Error;
use utils::lzw;

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
            data_root: get_app_dir(AppDataType::UserData, &APP_INFO, "UCI/optdigits").unwrap(),
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
    training_file: path::PathBuf,
    testing_file: path::PathBuf,
    info_file: path::PathBuf,
}

impl DataSetLoader {
    /// new
    pub fn new<P: AsRef<path::Path>>(data_path: P, download: bool) -> Result<DataSetLoader, Error> {
        let data_path = data_path.as_ref();
        fs::create_dir_all(data_path)?;

        let training_file = data_path.join("optdigits-orig.tra.Z");
        let testing_file = data_path.join("optdigits-orig.cv.Z");
        let info_file = data_path.join("optdigits-orig.names");

        if download {
            assure_file(&training_file, "http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits-orig.tra.Z")?;
            assure_file(&testing_file, "http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits-orig.cv.Z")?;
            assure_file(&info_file, "http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits-orig.names")?;
        }

        Ok(DataSetLoader{
            training_file,
            testing_file,
            info_file,
        })
    }

    pub fn load_training_data(&self) -> Result<Data, Error> {
        self.load_data(&self.training_file)
    }

    pub fn load_testing_data(&self) -> Result<Data, Error> {
        self.load_data(&self.testing_file)
    }

    pub fn load_info(&self) -> Result<String, Error> {
        let mut file = fs::File::open(&self.info_file)?;

        let mut info = String::new();
        file.read_to_string(&mut info)?;

        Ok(info)
    }

    fn load_data(&self, file: &path::Path) -> Result<Data, Error> {
        let input = lzw::Decoder::open(file)?;

        let mut line_count = 1;
        let data: Vec<_> = input
            // iterate over all bytes in the input
            .bytes()
            // panic on error
            .map(|c| c.unwrap())
            // count lines and skip certain characters
            .filter_map(|c| {
                match c {
                    b'\n' => {
                        line_count += 1;
                        None
                    }
                    b' ' => None,
                    _ => Some((c, line_count))
                }
            })
            // skip 21 header lines
            .skip_while(|&(_, line)| line < 22)
            // convert ASCII to numbers
            .map(|(c, line)| match c {
                b'0'...b'9' => c - b'0',
                _ => panic!(format!("Invalid character '{}' in data file (line {})", c as char, line))
            })
            .collect();

        Ok(Data::from(data))
    }
}

/// In-memory representation of the data
pub struct Data {
    data: Vec<u8>,
    n_samples: usize,
}

impl Data {
    fn from(data: Vec<u8>) -> Self {
        Data {
            n_samples: data.len() / (32 * 32 + 1),
            data,
        }
    }

    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    pub fn get_sample(&self, idx: usize) -> (ArrayView2<u8>, u8) {
        assert!(idx < self.n_samples);

        let start = (32 * 32 + 1) * idx;

        let x = ArrayView2::from_shape((32, 32), &self.data[start..start+1024]).unwrap();
        let y = self.data[start+1024];
        (x, y)
    }
}

impl CanonicalData for Data {
    fn to_canonical(&self) -> (Array2<f64>, Array2<f64>) {
        let x8 = ArrayView2::from_shape((self.n_samples, 1024).strides((1025, 1)), &self.data).unwrap();
        let y8 = ArrayView2::from_shape((self.n_samples, 1).strides((1025, 1)), &self.data[1024..]).unwrap();

        let mut x = Array2::zeros((self.n_samples, 1024));
        let mut y = Array2::zeros((self.n_samples, 1));

        Zip::from(&mut x).and(&x8).apply(|out, &inp| *out = inp as f64);
        Zip::from(&mut y).and(&y8).apply(|out, &inp| *out = inp as f64);

        (x, y)
    }
}


#[cfg(test)]
mod tests {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use super::*;

    #[test]
    fn load() {
        let data = DataSet::new().download(false).create().unwrap();
        let tst = data.load_testing_data().unwrap();
        assert_eq!(tst.n_samples, 946);
        // check class labels of a few specific samples
        assert_eq!(tst.get_sample(1).1, 6);
        assert_eq!(tst.get_sample(945).1, 5);


        let tra = data.load_training_data().unwrap();
        assert_eq!(tra.n_samples, 1934);
        // check class labels of a few specific samples
        assert_eq!(tra.get_sample(1).1, 0);
        assert_eq!(tra.get_sample(1933).1, 8);
    }

    fn checksum<'a, I: Iterator<Item=&'a f64>>(iter: I) -> u64 {
        let mut s = DefaultHasher::new();
        for &x in iter {
            (x as i64).hash(&mut s);
        }
        s.finish()
    }

    #[test]
    fn canonical() {
        let data = DataSet::new().download(false).create().unwrap();
        let (x_test, y_test) = data.load_testing_data().unwrap().into_canonical();
        assert_eq!(x_test.shape(), [946, 32 * 32]);
        assert_eq!(y_test.shape(), [946, 1]);
        assert_eq!(checksum(x_test.slice(s![42, ..]).iter()), 0xe65eee8be853c419);
        assert_eq!(y_test[[1, 0]], 6.0);
        assert_eq!(y_test[[945, 0]], 5.0);


        let  (x_train, y_train) = data.load_training_data().unwrap().into_canonical();
        assert_eq!(x_train.shape(), [1934, 32 * 32]);
        assert_eq!(y_train.shape(), [1934, 1]);
        assert_eq!(checksum(x_train.slice(s![42, ..]).iter()), 0xb75cb4f44968156d);
        assert_eq!(y_train[[1, 0]], 0.0);
        assert_eq!(y_train[[1933, 0]], 8.0);
    }
}
