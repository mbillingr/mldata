use std::io;

use app_dirs::AppDirsError;
use reqwest;

#[derive(Debug)]
pub enum Error {
    Io(io::Error),
    Download(reqwest::Error),
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