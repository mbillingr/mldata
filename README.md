# mldata
Rust crate for loading machine learning data sets.

### Goals
- Allowing easy access to popular machine learning data sets in rust.
- Load locally stored data but download on demand.
- Unified and consistent access to different types of data.

**Current status: in preparation**

## Usage
This crate is not yet published at [crates.io](https://crates.io/), so your `Cargo.toml` needs to link to the repository.
```toml
[dependencies]
mldata = { git = "https://github.com/mbillingr/mldata" }
```

```rust
extern crate mldata;

// Use the UCI Iris data set (https://archive.ics.uci.edu/ml/datasets/iris).
use mldata::uci_iris::DataSet;

fn main() {
  // Create a loader with default settings.
  // This will download the data when run for the first time.
  let loader = DataSet::new().create().unwrap();
  
  // Show some information about the data set.
  let info = loader.load_info().unwrap();
  println!("{}", info);
  
  // Load the data.
  let data = loader.load_data().unwrap();
  
  // Use the data.
  // Currently, the data API is somewhat limited. It only allows to access 
  // individual samples by index. This will be improved in the future.
  let sample = data.get_sample(42);
  println!("There are {} samples in tha iris data set.", data.n_samples());
  println!("Here is one of them:");
  println!("    Features: {:?}", sample.0);
  println!("       Class: {:?}", sample.1);
}
```

## Available Data Sets
- [Auto MPG](http://archive.ics.uci.edu/ml/datasets/Auto+MPG)
- [Iris](https://archive.ics.uci.edu/ml/datasets/iris)
- [Optical Recognition of Handwritten Digits](https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits)

## Cache
By default data is loaded from (and downloaded into) the user data directory. This has the advantage, that 
any applications using `mldata` share the same data directory and avoid unnecessary downloads. However, any 
directory can be specified instead, if the default behavior is not desired.

On Windows the data is placed in 
```path
<user directory>\AppData\Local\mldata
```

On Linux (Arch Linux in my particular case) the data is placed in
```path
~/.local/share/mldata/
```

## Help Wanted
The repository is hosted at [github](https://github.com/mbillingr/mldata). Issues and pull request welcome!

Contributions in the following areas are currently most welcome:
- All kind of pull requests, especiallly additional data sets.
- Suggestions on API design
- Suggestions how to unify access to diverse data sets (float, integer, categorical variables; classification/regression/etc.)

## License
MIT
