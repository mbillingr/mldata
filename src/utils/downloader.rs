//! Functions for downloading

use std::fs;
use std::io::{Read, Write};
use std::path;

use reqwest;

use utils::error::Error;

/// Make sure a file exists by downloading from given URL if necessary.
pub fn assure_file<P: AsRef<path::Path>, U: reqwest::IntoUrl>(file: P, url: U) -> Result<(), Error> {
    let file = file.as_ref();

    if !file.exists() {
        let mut file = fs::File::create(file)?;

        let mut content = reqwest::get(url)?;
        let mut data = Vec::new();
        content.read_to_end(&mut data)?;

        file.write_all(&data)?;
    }

    Ok(())
}
