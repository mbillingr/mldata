//! Routines for loading mldata.org data sets

use std::fs;

use app_dirs::*;

use utils::downloader::assure_file;
use utils::error::Error;
use utils::hdf5;

use common::APP_INFO;

fn load_mldata(name: &str, tables: &[&str]) -> Result<(), Error> {
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

    let filepath = data_root.join(filename + ".hdf5");

    assure_file(&filepath, &url)?;

    let file = hdf5::File::open(&filepath)?;

    for t in tables.iter() {
        let dset = file.dataset(t)?;
        println!("{:?}", dset.read());
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load() {
        //load_mldata("MNIST (original)").unwrap();
        load_mldata("uci-20070111 autoMpg", &["data/int0", "data/double1", "data/int2"]).unwrap();
    }
}
