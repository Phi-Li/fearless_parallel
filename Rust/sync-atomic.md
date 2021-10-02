# Module `std::sync::atomic`

Atomic types are the building blocks of other concurrent types.

Each method takes an `Ordering` which represents the strength of the memory barrier for that operation.

Atomic variables do not provide the mechanism for sharing and follow the threading model of Rust. The most common way to share an atomic variable is to put it into an `Arc` (an atomically-reference-counted shared pointer).

Atomic types may be stored in static variables, initialized using the constant initializers like `const fn new(v: bool) -> AtomicBool`. Atomic statics are often used for lazy global initialization.

## Portability

All atomic types in this module are guaranteed to be [_lock-free_](https://en.wikipedia.org/wiki/Non-blocking_algorithm). They don’t internally acquire a global mutex.

Atomic types and operations are not guaranteed to be _wait-free_. Operations like `fetch_or` may be implemented with a compare-and-swap loop.

Atomic operations may be implemented at the instruction layer with larger-size atomics, e.g., some platforms use 4-byte atomic instructions to implement `AtomicI8`.

- PowerPC and MIPS platforms with 32-bit pointers do not have `AtomicU64` or `AtomicI64` types.
- ARM platforms like `armv5te` that aren’t for Linux only provide `load` and `store` operations, and do not support Compare and Swap (CAS) operations, such as `swap`, `fetch_add`, etc.<br>
    Additionally on Linux, these CAS operations are implemented via operating system support, which may come with a performance penalty.
- ARM targets with `thumbv6m` only provide `load` and `store` operations, and do not support Compare and Swap (CAS) operations, such as `swap`, `fetch_add`, etc.

`AtomicUsize` and `AtomicIsize` are generally the most portable

For reference, the `std` library requires pointer-sized atomics, although `core` does not.

`#[cfg(target_arch)]` / `#[cfg(target_has_atomic)]`, conditional compilation.

## Examples

A simple spinlock:

```rust
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::{hint, thread};

fn main() {
    let spinlock = Arc::new(AtomicUsize::new(1));

    let spinlock_clone = Arc::clone(&spinlock);
    let thread = thread::spawn(move|| {
        spinlock_clone.store(0, Ordering::SeqCst);
    });

    // Wait for the other thread to release the lock
    while spinlock.load(Ordering::SeqCst) != 0 {
        hint::spin_loop();
    }

    if let Err(panic) = thread.join() {
        println!("Thread had an error: {:?}", panic);
    }
}
```

Keep a global count of live threads:

```rust
use std::sync::atomic::{AtomicUsize, Ordering};

static GLOBAL_THREAD_COUNT: AtomicUsize = AtomicUsize::new(0);

let old_thread_count = GLOBAL_THREAD_COUNT.fetch_add(1, Ordering::SeqCst);
println!("live threads: {}", old_thread_count + 1);
```

## Structs

`AtomicU64`

### Methods (17)

1. 
    ```rust
    pub const fn new(v: usize) -> AtomicUsize
    ```

---------------------------------------

2. 
    ```rust
    pub fn get_mut(&mut self) -> &mut usize
    ```

3. 
    ```rust
    pub fn into_inner(self) -> usize
    ```

---------------------------------------

4. 
    ```rust
    pub fn load(&self, order: Ordering) -> usize
    ```

5. 
    ```rust
    pub fn store(&self, val: usize, order: Ordering)
    ```

6. 
    ```rust
    pub fn swap(&self, val: usize, order: Ordering) -> usize
    ```

7. 
    ```rust
    pub fn compare_exchange(
        &self,
        current: usize,
        new: usize,
        success: Ordering,
        failure: Ordering
    ) -> Result<usize, usize>
    ```
    __Examples__
    ```rust
    let some_var = AtomicUsize::new(5);
    assert_eq!(some_var.compare_exchange(5, 10,
                                         Ordering::Acquire,
                                         Ordering::Relaxed),
               Ok(5));
    // `some_var` is 10
    assert_eq!(some_var.compare_exchange(6, 12,
                                         Ordering::SeqCst,
                                         Ordering::Acquire),
               Err(10));
    ```

8. 
    ```rust
    pub fn compare_exchange_weak(
        &self,
        current: usize,
        new: usize,
        success: Ordering,
        failure: Ordering
    ) -> Result<usize, usize>
    ```
    Result in more efficient code on some platforms.

    See [Crust of Rust: Atomics and Memory Ordering - YouTube](https://youtu.be/rMGWeSjctlY?t=3080) and [Load-link/store-conditional (LL/SC) - Wikipedia](https://en.wikipedia.org/wiki/Load-link/store-conditional)

    __Examples__
    ```rust
    let val = AtomicUsize::new(4);

    let mut old = val.load(Ordering::Relaxed);
    loop {
        let new = old * 2;
        match val.compare_exchange_weak(old, new, Ordering::SeqCst, Ordering::Relaxed) {
            Ok(_) => break,
            Err(x) => old = x,
        }
    }
    ```

---------------------------------------

9. 
    ```rust
    pub fn fetch_add(&self, val: usize, order: Ordering) -> usize
    ```

10. 
    ```rust
    pub fn fetch_sub(&self, val: usize, order: Ordering) -> usize
    ```

11. 
    ```rust
    pub fn fetch_and(&self, val: usize, order: Ordering) -> usize
    ```

12. 
    ```rust
    pub fn fetch_nand(&self, val: usize, order: Ordering) -> usize
    ```

13. 
    ```rust
    pub fn fetch_or(&self, val: usize, order: Ordering) -> usize
    ```

14. 
    ```rust
    pub fn fetch_xor(&self, val: usize, order: Ordering) -> usize
    ```

---------------------------------------

15. 
    ```rust
    pub fn fetch_update<F>(
        &self,
        set_order: Ordering,
        fetch_order: Ordering,
        f: F
    ) -> Result<usize, usize>
    where
        F: FnMut(usize) -> Option<usize>,
    ```

16. 
    ```rust
    pub fn fetch_max(&self, val: usize, order: Ordering) -> usize
    ```

17. 
    ```rust
    pub fn fetch_min(&self, val: usize, order: Ordering) -> usize
    ```

## Enums

`Ordering`: Atomic memory orderings

```rust
#[non_exhaustive]
pub enum Ordering {
    Relaxed,
    Release,
    Acquire,
    AcqRel,
    SeqCst,
}
```

Variants (Non-exhaustive)

- `Relaxed`: No ordering constraints, only atomic operations.

    Only the memory directly touched by the operation is synchronized.

- `Release`: store

    Notice: `Release` load and store = `Relaxed` load + `Release` store

- `Acquire`: load

    Notice: `Acquire` load and store = `Acquire` load + `Relaxed` store

- `AcqRel`: `Acquire` load && `Release` store

    Notice: `compare_and_swap`, it is possible that the operation ends up not performing any store and hence it has just `Acquire` ordering.

- `SeqCst`: load, store, load-with-store

    All threads see all sequentially consistent operations in the same order, preserving a total order of such operations across all threads.

## Functions

### `compiler_fence`: A compiler memory fence

`compiler_fence` does not emit any machine code, but restricts the kinds of memory re-ordering the compiler is allowed to do. The compiler may be disallowed from moving reads or writes from before or after the call to `compiler_fence` to the other side of the call.

- `Release`, preceding reads and writes cannot be moved __past__ subsequent writes.
- `Acquire`, subsequent reads and writes cannot be moved __ahead__ of preceding reads.
- `AcqRel`, both of the above rules are enforced.
- `SeqCst`, no re-ordering of reads and writes across this point is allowed.

`compiler_fence` is generally only useful for preventing a thread from _racing with itself_. In traditional programs, this can only occur when a signal handler is registered. In more low-level code, such situations can also arise when handling interrupts, when implementing green threads with _pre-emption_, etc.

__Examples__

```rust
use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::sync::atomic::Ordering;
use std::sync::atomic::compiler_fence;

static IMPORTANT_VARIABLE: AtomicUsize = AtomicUsize::new(0);
static IS_READY: AtomicBool = AtomicBool::new(false);

fn main() {
    IMPORTANT_VARIABLE.store(42, Ordering::Relaxed);
    // prevent earlier writes from being moved beyond this point
    compiler_fence(Ordering::Release);
    IS_READY.store(true, Ordering::Relaxed);
}

fn signal_handler() {
    if IS_READY.load(Ordering::Relaxed) {
        assert_eq!(IMPORTANT_VARIABLE.load(Ordering::Relaxed), 42);
    }
}
```

The compiler is free to swap the stores to `IMPORTANT_VARIABLE` and `IS_READY` since they are both `Ordering::Relaxed`. If it does, and the signal handler is invoked right after `IS_READY` is updated, then the signal handler will see `IS_READY=1`, but `IMPORTANT_VARIABLE=0`.

### `fence`: An atomic fence

A fence prevents the compiler and CPU from reordering certain types of memory operations around it. That creates synchronizes-with relationships between <u>it</u> and <u>atomic operations</u> or <u>fences</u> in other threads.

A fence which has `SeqCst` ordering, in addition to having both `Acquire` and `Release` semantics, participates in the global program order of the other `SeqCst` operations and/or fences.

Atomic operations with `Release` or `Acquire` semantics can also synchronize with a fence.

__Examples__

```rust
use std::sync::atomic::AtomicBool;
use std::sync::atomic::fence;
use std::sync::atomic::Ordering;

// A mutual exclusion primitive based on spinlock.
pub struct Mutex {
    flag: AtomicBool,
}

impl Mutex {
    pub fn new() -> Mutex {
        Mutex {
            flag: AtomicBool::new(false),
        }
    }

    pub fn lock(&self) {
        // Wait until the old value is `false`.
        while self
            .flag
            .compare_exchange_weak(false, true, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
        {}
        // This fence synchronizes-with store in `unlock`.
        fence(Ordering::Acquire);
    }

    pub fn unlock(&self) {
        self.flag.store(false, Ordering::Release);
    }
}
```
