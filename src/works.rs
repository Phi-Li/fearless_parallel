use std::collections;
use std::fs;
use std::io::{self, BufRead};


pub fn nsmallest<F>(file: &String, n: u64, keyfilter: F) -> Vec<(u64, u64)>
where
    F: Fn(u64) -> bool
{
    let mut heap = collections::BinaryHeap::new();
    let mut rd = io::BufReader::new(fs::File::open(file).unwrap());
    let l_buf = &mut String::new();

    while (heap.len() as u64) < n {
        // https://stackoverflow.com/questions/45220448
        l_buf.clear();
        if let Ok(0) = rd.read_line(l_buf) {
            break;
        } else {
            let pair: Vec<_> = l_buf.split(' ').map(
                |s| s.trim().parse().unwrap()
            ).collect();
            if keyfilter(pair[0]) {
                heap.push((pair[1], pair[0]));
            }
        }
    }

    l_buf.clear();
    while let Ok(num_bytes) = rd.read_line(l_buf) {
        if num_bytes == 0 {
            break;
        } else {
            let pair: Vec<u64> = l_buf.split(' ').map(
                |s| s.trim().parse().unwrap()
            ).collect();
            // Ignore duplicate values with different keys here (TODO)
            if keyfilter(pair[0]) && pair[1] < heap.peek().unwrap().0 {
                heap.pop();
                heap.push((pair[1], pair[0]));
            }
        }
        l_buf.clear();
    }

    heap.into_vec()
}