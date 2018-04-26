use std::fs;
use std::collections::{VecDeque, HashMap};
use std::io;
use std::io::Read;
use std::path::Path;


const CLEAR_TABLE: usize = 256;
const MAX_CODESIZE: usize = 16;


/// Read stream of integers with arbitrary bit length.
///
/// This is one horribly inefficient implementation of a bit stream. It's purpose is to get the job done for now.
struct BitReader<R> {
    input: R,
    buffer: VecDeque<u8>,
}

impl<R: Read> BitReader<R> {
    fn new(input: R) -> Self {
        BitReader {
            input,
            buffer: VecDeque::new(),
        }
    }

    fn get(&mut self, n_bits: usize) -> io::Result<Option<usize>> {
        assert!(n_bits <= 64);

        let mut buf = [0; 1];

        while self.buffer.len() < n_bits {
            let n = self.input.read(&mut buf)?;

            if n == 0 {
                return Ok(None)
            }

            let b = buf[0];
            self.buffer.push_back((b & 0b00000001) >> 0);
            self.buffer.push_back((b & 0b00000010) >> 1);
            self.buffer.push_back((b & 0b00000100) >> 2);
            self.buffer.push_back((b & 0b00001000) >> 3);
            self.buffer.push_back((b & 0b00010000) >> 4);
            self.buffer.push_back((b & 0b00100000) >> 5);
            self.buffer.push_back((b & 0b01000000) >> 6);
            self.buffer.push_back((b & 0b10000000) >> 7);
        }

        let mut result = 0;
        for i in 0..n_bits {
            result += (self.buffer.pop_front().unwrap() as usize) << i;
        }

        Ok(Some(result))
    }
}

/// A LZW decoder, or decompressor
///
/// This structure implements a [`std::io::Read`] interface and takes a stream of compressed data
/// as input, providing the decompressed data when read from.
///
/// # Examples
///
/// ```
/// fn main() {
///     use std::io::prelude::*;
///     use mldata::lzw::Decoder;
///
///     let data: &[u8] = &[0x61, 0xC4, 0x8C, 0x21];
///     let mut uncompressor = Decoder::new(data);
///     let mut s = String::new();
///     uncompressor.read_to_string(&mut s);
///     println!("{}", s);
///     assert_eq!(s, "abc");
/// }
/// ```
pub struct Decoder<R> {
    input: BitReader<R>,
    sequence_table: HashMap<usize, Vec<u8>>,
    buffer: VecDeque<u8>,
    previous_sequence: Vec<u8>,
    current_codesize: usize,
    next_code: usize,
}

impl Decoder<io::BufReader<fs::File>> {
    /// Create a new decoder which will decompress data read from the given file.
    ///
    /// This function works with files created by the unix tool `compress` (typically `.Z`
    /// extension).
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let file = fs::File::open(path)?;
        let mut reader = io::BufReader::new(file);

        let mut header = [0u8; 3];
        reader.read(&mut header)?;
        if header != [0x1f, 0x9d, 0x90] {
            return Err(io::Error::from(io::ErrorKind::InvalidData));
        }

        Ok(Decoder::new(reader))
    }
}

impl<R: Read> Decoder<R> {
    /// Create a new decoder which will decompress data read from the given stream.
    pub fn new(input: R) -> Self {
        let mut dec = Decoder {
            input: BitReader::new(input),
            sequence_table: HashMap::with_capacity(512),
            buffer: VecDeque::new(),
            previous_sequence: Vec::new(),
            current_codesize: 0,
            next_code: 0,
        };
        dec.reset();
        dec
    }

    fn reset(&mut self) {
        self.sequence_table.clear();
        for i in 0..256 {
            self.sequence_table.insert(i, vec![i as u8]);
        }
        self.sequence_table.insert(CLEAR_TABLE, Vec::new());
        self.previous_sequence.clear();
        self.current_codesize = 9;
        self.next_code = CLEAR_TABLE + 1;
    }

    fn advance_buffer(&mut self) -> bool {
        let code = match self.input.get(self.current_codesize).unwrap() {
            None => return false,
            Some(c) => c,
        };

        if code == CLEAR_TABLE {
            self.reset();
        }

        let sequence = if code == self.next_code {
            let mut tmp = self.previous_sequence.clone();
            tmp.push(self.previous_sequence[0]);
            self.sequence_table.insert(code, tmp.clone());
            tmp
        } else {
            self.sequence_table[&code].clone()
        };

        if !self.previous_sequence.is_empty() {
            self.previous_sequence.push(sequence[0]);
            self.sequence_table.insert(self.next_code, self.previous_sequence.clone());
            self.next_code += 1;

            if self.current_codesize < MAX_CODESIZE && (self.next_code >= 1 << self.current_codesize) {
                self.current_codesize += 1;
            }
        }

        self.previous_sequence = sequence.clone();
        self.buffer.extend(sequence);
        true
    }
}

impl<R: Read> Read for Decoder<R> {
    fn read(&mut self, buf: &mut[u8]) -> io::Result<usize> {
        while self.buffer.len() < buf.len() {
            if !self.advance_buffer() {
                break
            }
        }

        let n = self.buffer.len().min(buf.len());

        for (out, _) in buf.iter_mut().zip(0..n) {
            *out = self.buffer.pop_front().unwrap();
        }

        Ok(n)
    }
}

#[cfg(test)]
mod tests {
    use std::io::prelude::*;
    use super::*;

    #[test]
    fn bit_stream() {
        let mut bs = BitReader::new(&[42u8] as &[_]);
        assert_eq!(bs.get(9).unwrap(), None);

        let mut bs = BitReader::new(&[0u8, 85, 170, 255] as &[_]);
        assert_eq!(bs.get(8).unwrap(), Some(0));
        assert_eq!(bs.get(8).unwrap(), Some(85));
        assert_eq!(bs.get(8).unwrap(), Some(170));
        assert_eq!(bs.get(8).unwrap(), Some(255));
        assert_eq!(bs.get(1).unwrap(), None);

        let mut bs = BitReader::new(&[255u8, 255, 255, 255] as &[_]);
        assert_eq!(bs.get(2).unwrap(), Some(3));
        assert_eq!(bs.get(5).unwrap(), Some(31));
        assert_eq!(bs.get(8).unwrap(), Some(255));
        assert_eq!(bs.get(12).unwrap(), Some(4095));
        assert_eq!(bs.get(5).unwrap(), Some(31));
        assert_eq!(bs.get(1).unwrap(), None);

        let mut bs = BitReader::new(&[255u8, 0, 127, 128, 0, 0] as &[_]);
        assert_eq!(bs.get(12).unwrap(), Some(0b0000_11111111));
        assert_eq!(bs.get(12).unwrap(), Some(0b01111111_0000));
        assert_eq!(bs.get(12).unwrap(), Some(0b0000_10000000));
        assert_eq!(bs.get(12).unwrap(), Some(0b00000000_0000));
        assert_eq!(bs.get(1).unwrap(), None);
    }

    fn check_file(f: &str, expected: &str) {
        let mut dec = Decoder::open(f).expect(&format!("Could not open {}", f));
        let mut result = String::new();
        dec.read_to_string(&mut result).expect(&format!("Error while loading {}", f));
        assert_eq!(result, expected)
    }

    #[test]
    fn decoder() {
        check_file("data/abcdefg.txt.Z", "abcdefg\n");
        check_file("data/abababab.txt.Z", "abababab\n");
        check_file("data/xyz.txt.Z", "xyzxyzxyzxyzxyzxyzxyzxyzxyzxyzxyzxyz\n");
        check_file("data/0000000000.txt.Z", "0000000000\n");
        check_file("data/00000000001111100000.txt.Z", "00000000001111100000\n");

        let mut five = String::new();
        fs::File::open("data/5.txt").unwrap().read_to_string(&mut five).unwrap();
        check_file("data/5.txt.Z", &five);
    }
}
