# Module `std::sync`

## The __need__ for synchronization

```rust
static mut A: u32 = 0;
static mut B: u32 = 0;
static mut C: u32 = 0;

fn main() {
    unsafe {
        A = 3;
        B = 4;
        A = A + B;
        C = B;
        println!("{} {} {}", A, B, C);
        C = A;
    }
}
```

- Compiler optimizations
    - &emsp;
        ```rust
        // ...
        C = 4;
        A = 3;
        B = 4;
        // ...
        ```
    - `A = A + B`, lazy evaluation, with the global variable never getting updated.
    - [_Constant folding_](https://en.wikipedia.org/wiki/Constant_folding)
        ```rust
        println!("7 4 4");
        ```
- Concurrent execution

Note: Thanks to Rust’s safety guarantees, accessing global (static) variables requires `unsafe` code, assuming we don’t use any of the synchronization primitives in this module.

## Out-of-order execution

- The __compiler__ reordering instructions: If the compiler can issue an instruction at an earlier point, it will try to do so, <br>
e.g., it might hoist memory loads at the top of a code block, so that the CPU can start [_prefetching_](https://en.wikipedia.org/wiki/Cache_prefetching) the values from memory.<br>
In single-threaded scenarios, this can cause issues when writing signal handlers or certain kinds of low-level code.<br>
Use _compiler fences_ (`fn compiler_fence()`) to prevent this reordering.
- A __single processor__ executing instructions [_out-of-order_](https://en.wikipedia.org/wiki/Out-of-order_execution): [_superscalar_](https://en.wikipedia.org/wiki/Superscalar_processor) execution.<br>
    This kind of reordering is handled transparently by the CPU.
- A __multiprocessor__ system executing multiple hardware threads at the same time:
    - memory fences (`fn fence()`): ensure memory accesses are made visible to other CPUs in the right order.
    - atomic operations (Module `sync::atomic`): ensure simultaneous access to the same memory location doesn’t lead to undefined behavior.

## Higher-level synchronization objects

Most of the low-level synchronization primitives are quite error-prone and inconvenient to use.

For efficiency, the sync objects in the standard library are usually implemented with help from the operating system’s kernel, which is able to reschedule the threads while they are blocked on acquiring a lock.

- `mpsc`: Multi-producer, single-consumer queues, message-based communication.
- `Arc`: Atomically Reference-Counted pointer.
- `Mutex`: Mutual Exclusion mechanism.
- `RwLock`: Mutual exclusion mechanism which allows multiple readers at the same time, while allowing only one writer at a time. More efficient than `Mutex` in some cases.
- `Condvar`: Condition Variable, blocking a thread while waiting for an event to occur.
- `Barrier`: Ensures multiple threads will wait for each other to reach a point in the program, before continuing execution all together.
- `Once`: Thread-safe, one-time initialization of a global variable.

## Structs

### `Arc`: ‘Atomically Reference Counted’. A thread-safe reference-counting pointer.

If you need to mutate through an `Arc`, use `Mutex`, `RwLock`, or one of the `Atomic` types.

#### Thread Safety

Q: Why can’t you put a non-thread-safe type `T` in an `Arc<T>` to make it thread-safe?

A: If `Arc<T>` was always Send, `Arc<RefCell<T>>` would be as well. `RefCell<T>` isn’t `Sync`, not thread safe; it keeps track of the borrowing count using non-atomic operations.

In the end, this means that you may need to pair `Arc<T>` with some sort of `std::sync` type, usually `Mutex<T>`.

#### Breaking cycles with `Weak`

`Weak` pointers do not keep the __value__ inside the allocation alive; however, they do keep the __allocation__ (the backing store for the value) alive.

A tree could have strong `Arc` pointers from parent nodes to children, and `Weak` pointers from children back to their parents.

#### Examples

Sharing a mutable `AtomicUsize`:
```rust
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;

let val = Arc::new(AtomicUsize::new(5));

for _ in 0..10 {
    let val = Arc::clone(&val);

    thread::spawn(move || {
        let v = val.fetch_add(1, Ordering::SeqCst);
        println!("{:?}", v);
    });
}
```

Listing 16-15: Using an `Arc<T>` to wrap the `Mutex<T>` to be able to share ownership across multiple threads, Rust book
```rust
use std::sync::{Arc, Mutex};
use std::thread;

fn main() {
    let counter = Arc::new(Mutex::new(0));
    let mut handles = vec![];

    for _ in 0..10 {
        let counter = Arc::clone(&counter);
        let handle = thread::spawn(move || {
            let mut num = counter.lock().unwrap();

            *num += 1;
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    println!("Result: {}", *counter.lock().unwrap());
}
```

### `Mutex`: A mutual exclusion primitive protecting shared data

This mutex will block threads waiting for the lock to become available.

#### Poisoning

A mutex is considered _poisoned_ whenever a thread panics while holding the mutex. Once a mutex is poisoned, all other threads are unable to access the data by default as it is likely _tainted_ (some invariant is not being upheld).

Most usage of a mutex will simply `unwrap()` these results, propagating panics among threads to ensure that a possibly invalid invariant is not witnessed.

The `PoisonError` type has an `into_inner` method which will return the guard that would have otherwise been returned on a successful lock. This allows access to the data, despite the lock being poisoned.

#### Examples

```rust
use std::sync::{Arc, Mutex};
use std::thread;
use std::sync::mpsc::channel;

const N: usize = 10;

// Spawn a few threads to increment a shared variable (non-atomically), and
// let the main thread know once all increments are done.
//
// Here we're using an Arc to share memory among threads, and the data inside
// the Arc is protected with a mutex.
let data = Arc::new(Mutex::new(0));

let (tx, rx) = channel();
for _ in 0..N {
    let (data, tx) = (Arc::clone(&data), tx.clone());
    thread::spawn(move || {
        // The shared state can only be accessed once the lock is held.
        // Our non-atomic increment is safe because we're the only thread
        // which can access the shared state when the lock is held.
        //
        // We unwrap() the return value to assert that we are not expecting
        // threads to ever fail while holding the lock.
        let mut data = data.lock().unwrap();
        *data += 1;
        if *data == N {
            tx.send(()).unwrap();
        }
        // the lock is unlocked here when `data` goes out of scope.
    });
}

rx.recv().unwrap();
```

Recover from a poisoned mutex:
```rust
use std::sync::{Arc, Mutex};
use std::thread;

let lock = Arc::new(Mutex::new(0_u32));
let lock2 = Arc::clone(&lock);

let _ = thread::spawn(move || -> () {
    // This thread will acquire the mutex first, unwrapping the result of
    // `lock` because the lock has not been poisoned.
    let _guard = lock2.lock().unwrap();

    // This panic while holding the lock (`_guard` is in scope) will poison
    // the mutex.
    panic!();
}).join();

// The lock is poisoned by this point, but the returned result can be
// pattern matched on to return the underlying guard on both branches.
let mut guard = match lock.lock() {
    Ok(guard) => guard,
    Err(poisoned) => poisoned.into_inner(),
};

*guard += 1;
```

Manually drop the mutex guard to unlock it sooner than the end of the enclosing scope.
```rust
use std::sync::{Arc, Mutex};
use std::thread;

const N: usize = 3;

let data_mutex = Arc::new(Mutex::new(vec![1, 2, 3, 4]));
let res_mutex = Arc::new(Mutex::new(0));

let mut threads = Vec::with_capacity(N);
(0..N).for_each(|_| {
    let data_mutex_clone = Arc::clone(&data_mutex);
    let res_mutex_clone = Arc::clone(&res_mutex);

    threads.push(thread::spawn(move || {
        let mut data = data_mutex_clone.lock().unwrap();
        // This is the result of some important and long-ish work.
        let result = data.iter().fold(0, |acc, x| acc + x * 2);
        data.push(result);
        drop(data);
        *res_mutex_clone.lock().unwrap() += result;
    }));
});

let mut data = data_mutex.lock().unwrap();
// This is the result of some important and long-ish work.
let result = data.iter().fold(0, |acc, x| acc + x * 2);
data.push(result);
// We drop the `data` explicitly because it's not necessary anymore and the
// thread still has work to do. This allow other threads to start working on
// the data immediately, without waiting for the rest of the unrelated work
// to be done here.
//
// It's even more important here than in the threads because we `.join` the
// threads after that. If we had not dropped the mutex guard, a thread could
// be waiting forever for it, causing a deadlock.
drop(data);
// Here the mutex guard is not assigned to a variable and so, even if the
// scope does not end after this line, the mutex is still released: there is
// no deadlock.
*res_mutex.lock().unwrap() += result;

threads.into_iter().for_each(|thread| {
    thread
        .join()
        .expect("The thread creating or execution failed !")
});

assert_eq!(*res_mutex.lock().unwrap(), 800);
```

#### Methods (7)
1. 
    ```rust
    pub fn new(t: T) -> Mutex<T>
    ```
2. 
    ```rust
    pub fn unlock(guard: MutexGuard<'_, T>)
    ```
3. 
    ```rust
    pub fn lock(&self) -> LockResult<MutexGuard<'_, T>>
    ```
4. 
    ```rust
    pub fn try_lock(&self) -> TryLockResult<MutexGuard<'_, T>>
    ```
5. 
    ```rust
    pub fn is_poisoned(&self) -> bool
    ```
6. 
    ```rust
    pub fn into_inner(self) -> LockResult<T>
    where
        T: Sized,
    ```
7. 
    ```rust
    pub fn get_mut(&mut self) -> LockResult<&mut T>
    ```

### `RwLock`: A reader-writer lock

A number of readers or at most one writer at any point in time.
- Write portion: exclusive access, allows modification of the underlying data;
- Read portion: shared access, allows for read-only access.

The priority policy of the lock is dependent on the underlying operating system’s implementation. In particular, a writer which is waiting to acquire the lock in `write` might or might not block concurrent calls to `read`.

#### Poisoning

Note: an `RwLock` may only be poisoned if a panic occurs while it is locked exclusively (write mode).

#### Examples

```rust
use std::sync::RwLock;

let lock = RwLock::new(5);

// many reader locks can be held at once
{
    let r1 = lock.read().unwrap();
    let r2 = lock.read().unwrap();
    assert_eq!(*r1, 5);
    assert_eq!(*r2, 5);
} // read locks are dropped at this point

// only one write lock may be held, however
{
    let mut w = lock.write().unwrap();
    *w += 1;
    assert_eq!(*w, 6);
} // write lock is dropped here
```

#### Methods (8)
1. 
    ```rust
    pub fn new(t: T) -> RwLock<T>
    ```
2. 
    ```rust
    pub fn read(&self) -> LockResult<RwLockReadGuard<'_, T>>
    ```
3. 
    ```rust
    pub fn try_read(&self) -> TryLockResult<RwLockReadGuard<'_, T>>
    ```
4. 
    ```rust
    pub fn write(&self) -> LockResult<RwLockWriteGuard<'_, T>>
    ```
5. 
    ```rust
    pub fn try_write(&self) -> TryLockResult<RwLockWriteGuard<'_, T>>
    ```
6. 
    ```rust
    pub fn is_poisoned(&self) -> bool
    ```
7. 
    ```rust
    pub fn into_inner(self) -> LockResult<T>
    where
        T: Sized,
    ```
8. 
    ```rust
    pub fn get_mut(&mut self) -> LockResult<&mut T>
    ```

### `Condvar`: A condition Variable

Condition variables are typically associated with a boolean predicate (a condition) and a mutex (`(Mutex::new(false), Condvar::new())`).

Functions in this module will block the __current thread__ of execution.

Note: any attempt to use multiple mutexes on the same condition variable may result in a runtime panic.

```rust
use std::sync::{Arc, Mutex, Condvar};
use std::thread;

let pair = Arc::new((Mutex::new(false), Condvar::new()));
let pair2 = Arc::clone(&pair);

// Inside of our lock, spawn a new thread, and then wait for it to start.
thread::spawn(move|| {
    let (lock, cvar) = &*pair2;
    let mut started = lock.lock().unwrap();
    *started = true;
    // We notify the condvar that the value has changed.
    cvar.notify_one();
});

// Wait for the thread to start up.
let (lock, cvar) = &*pair;
let mut started = lock.lock().unwrap();
while !*started {
    started = cvar.wait(started).unwrap();
}
```

#### Methods (7)
1. 
    ```rust
    pub fn new() -> Condvar
    ```
2. 
    ```rust
    pub fn wait<'a, T>(
        &self,
        guard: MutexGuard<'a, T>
    ) -> LockResult<MutexGuard<'a, T>>
    ```
3. 
    ```rust
    pub fn wait_while<'a, T, F>(
        &self,
        guard: MutexGuard<'a, T>,
        condition: F
    ) -> LockResult<MutexGuard<'a, T>>
    where
        F: FnMut(&mut T) -> bool,
    ```
4. 
    ```rust
    pub fn wait_timeout<'a, T>(
        &self,
        guard: MutexGuard<'a, T>,
        dur: Duration
    ) -> LockResult<(MutexGuard<'a, T>, WaitTimeoutResult)>
    ```
5. 
    ```rust
    pub fn wait_timeout_while<'a, T, F>(
        &self,
        guard: MutexGuard<'a, T>,
        dur: Duration,
        condition: F
    ) -> LockResult<(MutexGuard<'a, T>, WaitTimeoutResult)>
    where
        F: FnMut(&mut T) -> bool,
    ```
6. 
    ```rust
    pub fn notify_one(&self)
    ```
7. 
    ```rust
    pub fn notify_all(&self)
    ```

### `Barrier`: Multiple threads synchronize the beginning of some computation.

```rust
use std::sync::{Arc, Barrier};
use std::thread;

let mut handles = Vec::with_capacity(10);
let barrier = Arc::new(Barrier::new(10));
for _ in 0..10 {
    let c = Arc::clone(&barrier);
    // The same messages will be printed together.
    // You will NOT see any interleaving.
    handles.push(thread::spawn(move|| {
        println!("before wait");
        c.wait();
        println!("after wait");
    }));
}
// Wait for other threads to finish.
for handle in handles {
    handle.join().unwrap();
}
```

#### Methods (2)
1. 
    ```rust
    pub fn new(n: usize) -> Barrier
    ```
2. 
    ```rust
    pub fn wait(&self) -> BarrierWaitResult`
    ```

### `Once`: A synchronization primitive running a one-time global initialization. One-time initialization for FFI (foreign function interface) or related functionality.

```rust
use std::sync::Once;

static mut VAL: usize = 0;
static INIT: Once = Once::new();

// Accessing a `static mut` is unsafe much of the time, but if we do so
// in a synchronized fashion (e.g., write once or read all) then we're
// good to go!
//
// This function will only call `expensive_computation` once, and will
// otherwise always return the value returned from the first invocation.
fn get_cached_val() -> usize {
    unsafe {
        INIT.call_once(|| {
            VAL = expensive_computation();
        });
        VAL
    }
}

fn expensive_computation() -> usize {
    // ...
}
```

[singleton.rs · lpxxn/rust-design-pattern](https://github.com/lpxxn/rust-design-pattern/blob/master/creational/singleton.rs)

#### Methods (4)
1. 
    ```rust
    pub const fn new() -> Once
    ```
2. 
    ```rust
    pub fn call_once<F>(&self, f: F)
    where
        F: FnOnce(),
    ````
3. 
    ```rust
    pub fn call_once_force<F>(&self, f: F)
    where
        F: FnOnce(&OnceState),
    ```
4. 
    ```rust
    pub fn is_completed(&self) -> bool
    ```

## Modules

### `mpsc`: Multi-producer, single-consumer FIFO queue communication primitives.

1. An __asynchronous__, infinitely buffered channel. `(Sender, Receiver)` tuple, all sends will never block, conceptually infinite buffer.
2. A __synchronous__, bounded channel. `(SyncSender, Receiver)` tuple, the storage for pending messages is a pre-allocated buffer of a fixed size. All sends will be synchronous by blocking until there is buffer space available.<br>
    Note: a bound of 0 is allowed, causing the channel to become a “rendezvous” channel where each sender atomically hands off a message to a receiver.

Shared usage:
```rust
use std::thread;
use std::sync::mpsc::channel;

// Create a shared channel that can be sent along from many threads
// where tx is the sending half (tx for transmission), and rx is the receiving
// half (rx for receiving).
let (tx, rx) = channel();
for i in 0..10 {
    let tx = tx.clone();
    thread::spawn(move|| {
        tx.send(i).unwrap();
    });
}

for _ in 0..10 {
    let j = rx.recv().unwrap();
    assert!(0 <= j && j < 10);
}
```

Synchronous channels:
```rust
use std::thread;
use std::sync::mpsc::sync_channel;

let (tx, rx) = sync_channel::<i32>(0);
thread::spawn(move|| {
    // This will wait for the parent thread to start receiving
    tx.send(53).unwrap();
});
rx.recv().unwrap();
```

### `atomic`: Atomic types

Atomic types are the building blocks of other concurrent types.

Each method takes an `Ordering` which represents the strength of the memory barrier for that operation.

Atomic variables do not provide the mechanism for sharing and follow the threading model of Rust. The most common way to share an atomic variable is to put it into an `Arc` (an atomically-reference-counted shared pointer).

Atomic types may be stored in static variables, initialized using the constant initializers like `const fn new(v: bool) -> AtomicBool`. Atomic statics are often used for lazy global initialization.

#### Portability

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

#### Examples

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

#### Enums

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

#### Functions

##### `compiler_fence`: A compiler memory fence

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

##### `fence`: An atomic fence

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
