# Fearless parallelism

A simple parallel processing example in Rust.

The program will find out the smallest three values along with their keys in all of the input files.

Repo name is from chapter [Fearless Concurrency](https://doc.rust-lang.org/book/ch16-00-concurrency.html) in Rust book.

## Usage

```
$ cargo run
```

## TODO
- [ ] Replace unsafe `.unwrap()` with error handling
- [ ] CLI (interface)
- [ ] Unit test for works in `works.rs`
- [ ] Abstraction for divide and merge process
