use std;
use std::ffi::CString;
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
pub enum DynamicArray {
    Int8(Array<i8, IxDyn>),
    Int16(Array<i16, IxDyn>),
    Int32(Array<i32, IxDyn>),
    Int64(Array<i64, IxDyn>),
    UInt8(Array<u8, IxDyn>),
    UInt16(Array<u16, IxDyn>),
    UInt32(Array<u32, IxDyn>),
    UInt64(Array<u64, IxDyn>),
    Float32(Array<f32, IxDyn>),
    Float64(Array<f64, IxDyn>),
}

pub struct File {
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

pub struct Dataset {
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
            if datatype.equal_id(H5T_NATIVE_INT8) {
                let data = self.raw_read(H5T_NATIVE_INT8, size)?;
                let array = Array::from_shape_vec(IxDyn(&shape), data)?;
                Ok(DynamicArray::Int8(array))
            } else if datatype.equal_id(H5T_NATIVE_INT16) {
                let data = self.raw_read(H5T_NATIVE_INT16, size)?;
                let array = Array::from_shape_vec(IxDyn(&shape), data)?;
                Ok(DynamicArray::Int16(array))
            } else if datatype.equal_id(H5T_NATIVE_INT32) {
                let data = self.raw_read(H5T_NATIVE_INT32, size)?;
                let array = Array::from_shape_vec(IxDyn(&shape), data)?;
                Ok(DynamicArray::Int32(array))
            } else if datatype.equal_id(H5T_NATIVE_INT64) {
                let data = self.raw_read(H5T_NATIVE_INT64, size)?;
                let array = Array::from_shape_vec(IxDyn(&shape), data)?;
                Ok(DynamicArray::Int64(array))
            } else if datatype.equal_id(H5T_NATIVE_UINT8) {
                let data = self.raw_read(H5T_NATIVE_UINT8, size)?;
                let array = Array::from_shape_vec(IxDyn(&shape), data)?;
                Ok(DynamicArray::UInt8(array))
            } else if datatype.equal_id(H5T_NATIVE_UINT16) {
                let data = self.raw_read(H5T_NATIVE_UINT16, size)?;
                let array = Array::from_shape_vec(IxDyn(&shape), data)?;
                Ok(DynamicArray::UInt16(array))
            } else if datatype.equal_id(H5T_NATIVE_UINT32) {
                let data = self.raw_read(H5T_NATIVE_UINT32, size)?;
                let array = Array::from_shape_vec(IxDyn(&shape), data)?;
                Ok(DynamicArray::UInt32(array))
            } else if datatype.equal_id(H5T_NATIVE_UINT64) {
                let data = self.raw_read(H5T_NATIVE_UINT64, size)?;
                let array = Array::from_shape_vec(IxDyn(&shape), data)?;
                Ok(DynamicArray::UInt64(array))
            } else if datatype.equal_id(H5T_NATIVE_FLOAT) {
                let data = self.raw_read(H5T_NATIVE_FLOAT, size)?;
                let array = Array::from_shape_vec(IxDyn(&shape), data)?;
                Ok(DynamicArray::Float32(array))
            } else if datatype.equal_id(H5T_NATIVE_DOUBLE) {
                let data = self.raw_read(H5T_NATIVE_DOUBLE, size)?;
                let array = Array::from_shape_vec(IxDyn(&shape), data)?;
                Ok(DynamicArray::Float64(array))
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

pub struct Datatype {
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

pub struct Dataspace {
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
