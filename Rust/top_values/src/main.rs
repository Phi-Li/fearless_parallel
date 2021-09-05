use std::collections::{HashMap, BinaryHeap};
use std::env;
use std::sync;
use std::thread;

mod works;
use crate::works::nsmallest;

fn main() {
    const TOP_N: u64 = 3;
    let keyfilter = |x: u64| x >= 1 && x < 100;

    let args: Vec<String> = env::args().skip(1).collect();

    // Generate worker IDs, we simply use 0, 1, 2 ... here
    // let worker_ids;

    // Distribute specific tasks among workers, we use pre-divided text files here
    let mut task_map = HashMap::new();
    for (worker_i, path) in args.into_iter().enumerate() {
        task_map.insert(worker_i, sync::Arc::new(path));
    }

    // Launch workers
    let mut tpoll = HashMap::new();     // Act as similar role of epoll in Linux
    let (tx, rx) = sync::mpsc::channel();
    for (&id, input) in task_map.iter() {
        let input = sync::Arc::clone(input);
        let tx = sync::mpsc::Sender::clone(&tx);
        tpoll.insert(
            id,
            thread::spawn(
                move || {
                    println!("Worker {} start.", id);
                    let ret = nsmallest(&input, TOP_N, keyfilter);
                    tx.send(id).unwrap();
                    println!("Worker {} exit.", id);
                    ret
                }
            )
        );
    }
    drop(tx);

    // Merge results from workers
    let mut heap = BinaryHeap::new();
    for r in rx {
        let mut result = tpoll.remove(&r).unwrap().join().unwrap();

        if (heap.len() as u64) < TOP_N {
            while (heap.len() as u64) < TOP_N {
                if let Some(pair) = result.pop() {
                    heap.push(pair);
                } else {
                    break;
                }
            }
        } else {
            while let Some(pair) = result.pop() {
                // Ignore duplicate values with different keys here (TODO)
                if pair.0 < heap.peek().unwrap().0 {
                    heap.pop();
                    heap.push(pair);
                }
            }
        }
    }

    for &(v, k) in heap.iter() {
        println!("{}: {}", k, v);
    }
}