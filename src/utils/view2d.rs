//! 2D view into slices.

use std::fmt;
use std::fmt::Write;
use std::ops;

/// A contiguous 2D view into a 1D slice
pub struct View2D<'a, T:'a> {
    data: &'a [T],
    n_rows: usize,
    n_cols: usize,
}

impl<'a, T: 'a> View2D<'a, T> {
    pub fn new(data: &'a [T], n_rows: usize, n_cols: usize) -> Self {
        assert_eq!(n_rows * n_cols, data.len());
        View2D {
            data,
            n_rows,
            n_cols,
        }
    }
}

impl<'a, T: 'a> fmt::Debug for View2D<'a, T>
    where T: fmt::Debug
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_char('[')?;
        let mut i = self.data.iter();
        for r in 0..self.n_rows {
            for c in 0..self.n_cols {
                write!(f, "{:?}", i.next().unwrap())?;
                if c < self.n_cols - 1 {
                    write!(f, ", ")?;
                }
            }
            if r < self.n_rows - 1 {
                write!(f, "; ")?;
            }
        }
        f.write_char(']')?;
        Ok(())
    }
}

impl<'a, T: 'a> ops::Index<(usize, usize)> for View2D<'a, T> {
    type Output = T;
    fn index(&self, (r, c): (usize, usize)) -> &T {
        assert!(r < self.n_rows);
        assert!(c < self.n_cols);
        &self.data[c + r * self.n_cols]
    }
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_format() {
        let v = View2D::new(&[1, 2, 3, 4, 5, 6], 3, 2);
        assert_eq!(format!("{:?}", v), "[1, 2; 3, 4; 5, 6]");
    }

    #[test]
    #[should_panic]
    fn fail_init() {
        View2D::new(&[1, 2, 3, 4, 5, 6], 2, 2);
    }

    #[test]
    fn index() {
        let v = View2D::new(&[1, 2, 3, 42, 5, 6], 3, 2);
        assert_eq!(1, v[(0, 0)]);
        assert_eq!(3, v[(1, 0)]);
        assert_eq!(5, v[(2, 0)]);
        assert_eq!(2, v[(0, 1)]);
        assert_eq!(42, v[(1, 1)]);
        assert_eq!(6, v[(2, 1)]);
    }
}