//! Our error type.

use std::io;

use app_dirs::AppDirsError;
use reqwest;

use ndarray::ShapeError;

use utils::hdf5;

#[derive(Debug)]
pub enum Error {
    Io(io::Error),
    Download(reqwest::Error),
    Hdf5Error(hdf5::Error),
    ArrayError(ShapeError),
    DataType,
    Internal,
}

impl From<AppDirsError> for Error {
    fn from(err: AppDirsError) -> Error {
        match err {
            AppDirsError::Io(e) => Error::Io(e),
            _ => Error::Internal,
        }
    }
}

impl From<hdf5::Error> for Error {
    fn from(err: hdf5::Error) -> Error {
        match err {
            hdf5::Error::IoError(ioe) => Error::Io(ioe),
            _ => Error::Hdf5Error(err)
        }
    }
}

impl From<ShapeError> for Error {
    fn from(err: ShapeError) -> Error {
        Error::ArrayError(err)
    }
}

impl From<reqwest::Error> for Error {
    fn from(err: reqwest::Error) -> Error {
        Error::Download(err)
    }
}

impl From<io::Error> for Error {
    fn from(err: io::Error) -> Error {
        Error::Io(err)
    }
}