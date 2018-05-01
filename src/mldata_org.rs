//! Routines for loading mldata.org data sets

use std::fs;

use app_dirs::*;

use utils::downloader::assure_file;
use utils::error::Error;

use common::APP_INFO;

fn load_mldata(name: &str) -> Result<(), Error> {
    let filename: String = name.to_lowercase().chars().filter_map(|c| {
        match c {
            ' ' => Some('-'),
            '(' | ')' | '.' => None,
            _ => Some(c),
        }
    }).collect();
    let data_root = get_app_dir(AppDataType::UserData, &APP_INFO, "mldata.org")?;

    fs::create_dir_all(&data_root)?;

    let url = "http://mldata.org/repository/data/download/".to_owned() + &filename + "/";
    //let url = "http://mldata.org/repository/data/download/mnist-original/";

    let filepath = data_root.join(filename + ".hdf5");

    assure_file(&filepath, &url)?;

    //let f = File::open(&filepath, "r").unwrap();

    //println!("{:?}", f);

    hdf5::open(&filepath);

    Ok(())
}

mod hdf5 {
    use std;
    use std::ffi::{OsStr, CString};
    use std::path::Path;
    use std::ptr::null_mut;
    use std::result;
    use hdf5_sys::*;
    use ndarray::{Array, IxDyn, ShapeError};
    use num::traits::Zero;

    #[derive(Debug)]
    pub enum Error {
        IoError(std::io::Error),
        NdError(ShapeError),
        UnsupportedDataType,
        UnknownError,
    }

    impl From<ShapeError> for Error {
        fn from(err: ShapeError) -> Self {
            Error::NdError(err)
        }
    }

    pub type Result<T> = result::Result<T, Error>;

    #[derive(Debug)]
    enum DynamicArray {
        Int32(Array<i32, IxDyn>),
        Float32(Array<f32, IxDyn>),
    }

    struct File {
        id: hid_t,
    }

    impl File {
        pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
            let filename = path.as_ref().to_str().unwrap();
            let filename_c = CString::new(filename).unwrap();

            let id = unsafe {
                H5Fopen(filename_c.as_ptr(), H5F_ACC_RDONLY, H5P_DEFAULT)
            };

            if id < 0 {
                let msg = format!("File not found: {:?}", filename);
                let err = std::io::Error::new(std::io::ErrorKind::NotFound, msg);
                return Err(Error::IoError(err));
            }

            Ok(File{id})
        }

        pub fn dataset(&self, name: &str) -> Result<Dataset> {
            Dataset::new(self, name)
        }
    }

    impl Drop for File {
        fn drop(&mut self) {
            unsafe {
                H5Fclose(self.id);
            }
        }
    }

    struct Dataset {
        id: hid_t,
    }

    impl Dataset {
        pub fn new(file: &File, name: &str) -> Result<Self> {
            let name_c = CString::new(name).unwrap();

            let id = unsafe {
                H5Dopen2(file.id, name_c.as_ptr(), H5P_DEFAULT)
            };

            if id < 0 {
                let msg = format!("Dataset not found: {}", name);
                let err = std::io::Error::new(std::io::ErrorKind::NotFound, msg);
                return Err(Error::IoError(err));
            }

            Ok(Dataset {
                id
            })
        }

        pub fn get_type(&self) -> Datatype {
            Datatype::new(self)
        }

        pub fn get_space(&self) -> Dataspace {
            Dataspace::new(self)
        }

        unsafe fn raw_read<T: Zero + Copy>(&self, mem_type: hid_t, size: usize) -> Result<Vec<T>> {
            let mut data: Vec<T> = vec![T::zero(); size];
            if H5Dread(self.id, mem_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.as_mut_ptr() as *mut _) < 0 {
                Err(Error::UnknownError)
            } else {
                Ok(data)
            }
        }

        pub fn read(&self) -> Result<DynamicArray> {
            let datatype = self.get_type();
            let space = self.get_space();
            let shape = space.shape()?;

            let size = shape.iter().product();

            unsafe {
                if datatype.equal_id(H5T_NATIVE_INT32) {
                    let data = self.raw_read(H5T_NATIVE_INT32, size)?;
                    let array = Array::from_shape_vec(IxDyn(&shape), data)?;
                    Ok(DynamicArray::Int32(array))
                } else if datatype.equal_id(H5T_NATIVE_FLOAT) {
                    let data = self.raw_read(H5T_NATIVE_FLOAT, size)?;
                    let array = Array::from_shape_vec(IxDyn(&shape), data)?;
                    Ok(DynamicArray::Float32(array))
                } else {
                    Err(Error::UnsupportedDataType)
                }
            }
        }
    }

    impl Drop for Dataset {
        fn drop(&mut self) {
            unsafe {
                H5Dclose(self.id);
            }
        }
    }

    struct Datatype {
        id: hid_t,
    }

    impl Datatype {
        pub fn new(dset: &Dataset) -> Self {
            let id = unsafe {
                H5Dget_type(dset.id)
            };

            Datatype {
                id
            }
        }

        pub fn equal_id(&self, other: hid_t) -> bool {
            unsafe {
                H5Tequal(self.id, other) == 1
            }
        }
    }

    impl Drop for Datatype {
        fn drop(&mut self) {
            unsafe {
                H5Tclose(self.id);
            }
        }
    }

    struct Dataspace {
        id: hid_t,
    }

    impl Dataspace {
        pub fn new(dset: &Dataset) -> Self {
            let id = unsafe {
                H5Dget_space(dset.id)
            };

            Dataspace {
                id
            }
        }

        pub fn ndims(&self) -> Result<usize> {
            let n = unsafe {
                H5Sget_simple_extent_ndims(self.id)
            };

            if n < 0 {
                Err(Error::UnknownError)
            } else {
                Ok(n as usize)
            }
        }

        pub fn shape(&self) -> Result<Vec<usize>> {
            let ndims = self.ndims()?;
            let mut shape = vec![0; ndims as usize];

            let result = unsafe {
                H5Sget_simple_extent_dims(self.id, shape.as_mut_ptr() as *mut u64, null_mut())
            };

            if result < 0 || result as usize != ndims {
                return Err(Error::UnknownError)
            }

            Ok(shape)
        }
    }

    impl Drop for Dataspace {
        fn drop(&mut self) {
            unsafe {
                H5Sclose(self.id);
            }
        }
    }

    pub fn load_hdf5<P: AsRef<Path>>(path: P, name: &str) -> Result<()> {
        /*let filename = path.as_ref().to_str().unwrap();
        let filename_c = CString::new(filename).unwrap();*/
        let name_c = CString::new(name).unwrap();

        let file = File::open(path)?;

        let dset = file.dataset("/data/int0")?;

        let dtype = dset.get_type();

        let space = dset.get_space();

        println!("{:?}", dset.read());

        Ok(())
    }

    pub fn open<P: AsRef<Path>>(path: P) {
        load_hdf5(path, "/data/int0");
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load() {
        //load_mldata("MNIST (original)").unwrap();
        load_mldata("uci-20070111 autoMpg").unwrap();
    }
}
