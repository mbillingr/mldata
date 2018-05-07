//! Load (and download) machine learning data sets

extern crate app_dirs;
extern crate hdf5_sys;

// workaround to supress warning that macro_use is unused; it is used in some tests, though.
#[cfg(test)]
#[macro_use(s)]
extern crate ndarray;
#[cfg(not(test))]
extern crate ndarray;

extern crate num;

extern crate reqwest;

pub mod canonical;
pub mod common;
pub mod utils;

pub mod mldata_auto_mpg;
pub mod uci_auto_mpg;
pub mod uci_iris;
pub mod uci_optdigits;
