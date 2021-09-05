# Fearless parallelism

A simple parallel processing example in Rust.

The program will find out the smallest three values along with their keys in all of the input files.

Repo name is from chapter [Fearless Concurrency](https://doc.rust-lang.org/book/ch16-00-concurrency.html) in Rust book.

## Usage

```
$ cargo run --package fearless_parallel -- Rust/top_values/input_0.txt Rust/top_values/input_1.txt Rust/top_values/input_2.txt Rust/top_values/input_3.txt
```

## TODO
- [ ] Replace unsafe `.unwrap()` with error handling
- [X] CLI (interface)
- [ ] Unit test for works in `works.rs`
- [ ] Abstraction for divide and merge process
