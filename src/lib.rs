//! Load (and download) machine learning data sets

extern crate app_dirs;
#[macro_use(s)]
extern crate ndarray;
extern crate reqwest;

pub mod canonical;
pub mod common;
pub mod utils;

pub mod uci_iris;
pub mod uci_optdigits;
