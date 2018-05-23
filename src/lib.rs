//! Load (and download) machine learning data sets

extern crate app_dirs;
extern crate arff;

// workaround to suppress warning that macro_use is unused; it is used in some tests, though.
#[cfg(test)]
#[macro_use(s)]
extern crate ndarray;
#[cfg(not(test))]
extern crate ndarray;

extern crate num;
extern crate reqwest;
extern crate serde;

#[macro_use]
extern crate serde_derive;

pub mod canonical;
pub mod common;
pub mod utils;

pub mod openml;

pub mod uci_auto_mpg;
pub mod uci_iris;
pub mod uci_optdigits;
