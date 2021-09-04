use std::collections::{HashMap, BinaryHeap};
use std::sync;
use std::thread;

mod works;
use crate::works::nsmallest;

fn main() {
    let WORKER_NUM: u8 = 4;
    let N: u64 = 3;
    let keyfilter = |x: u64| x >= 1 && x < 100;

    // Generate worker IDs, we simply use 0, 1, 2 ... here
    let worker_ids: Vec<u8> = (0..WORKER_NUM).map(|x| x).collect();

    // Distribute specific tasks among workers, we use pre-divided text files here
    let mut task_map = HashMap::new();
    for &id in worker_ids.iter() {
        task_map.insert(id, sync::Arc::new(format!("input_{}.txt", id)));
    }

    // Launch workers
    let mut tpoll = HashMap::new();     // Act as similar role of epoll in Linux
    let (tx, rx) = sync::mpsc::channel();
    for (&id,input) in task_map.iter() {
        let input = sync::Arc::clone(input);
        let tx = sync::mpsc::Sender::clone(&tx);
        tpoll.insert(
            id,
            thread::spawn(
                move || {
                    println!("Worker {} start.", id);
                    let ret = nsmallest(&input, N, keyfilter);
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

        if (heap.len() as u64) < N {
            while (heap.len() as u64) < N {
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