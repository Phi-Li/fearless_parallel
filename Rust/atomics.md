# Atomics

Rust pretty blatantly just inherits the memory model for atomics from C++20, which is a pragmatic concession to the fact that _everyone_ is pretty bad at modeling atomics. At very least, we can benefit from existing tooling and research around the C/C++ memory model.

(You'll often see this model referred to as "C/C++11" or just "C11". C just copies the C++ memory model; and C++11 was the first version of the model but it has received some bugfixes since then.)

The C++ memory model is fundamentally about trying to bridge the gap between the semantics _we_ want, the optimizations _compilers_ want, and the inconsistent chaos our _hardware_ wants.

## Compiler Reordering

```rust
x = 1;
y = 3;
x = 2;
```

- Inverted the order of events
- Eliminated one event completely

```rust
x = 2;
y = 3;
```

- Single-threaded: Completely unobservable, after all the statements have executed we are in exactly the same state.
- Multi-threaded: We may have been relying on `x = 1` before `y = 3`.

## Hardware Reordering

```
initial state: x = 0, y = 1

1. y = 3
THREAD 1        THREAD2
y = 3;
                if x == 1 {
x = 1;              y *= 2;
                }

2. y = 6
THREAD 1        THREAD2
y = 3;
x = 1;
                if x == 1 {
                    y *= 2;
                }

3. y = 2
THREAD 1        THREAD2
y = _;
x = 1;
(y <= 3)        if x == 1 {
                    y *= 2;
                }
```

CPU guarantees:
- Strongly-ordered: Most notably x86/64
    - Asking for strong guarantees: Cheap or even free.
    - Asking for weak guarantees: More likely to _happen_ to work, even though your program is strictly incorrect.
- Weakly-ordered: ARM
    - Asking for weak guarantees: Performance wins.

If possible, concurrent algorithms should be tested on weakly-ordered hardware.

## Data Accesses and atomic accesses

The way we communicate _happens before_ relationships are through _data accesses_ and _atomic accesses_.

__Data accesses__

Data accesses are free to be reordered by the compiler on the assumption that the program is single-threaded.

The hardware is also free to propagate the changes made in data accesses to other threads as lazily and inconsistently as it wants.

Most critically, data accesses are how data races happen.

__Atomic accesses__

Each atomic access can be marked with an _ordering_ that specifies what kind of relationship it establishes with other accesses.

In practice, this boils down to telling the compiler and hardware certain things they can't do.

- Compiler: Re-ordering of instructions.
- Hardware: How writes are propagated to other threads.

## Sequentially Consistent

Sequentially Consistent is the most powerful of all, implying the restrictions of all other orderings. Intuitively, a sequentially consistent operation cannot be reordered: all accesses on one thread that happen before and after a `SeqCst` access stay __before__ and __after__ it.

A data-race-free program that uses only sequentially consistent atomics and data accesses has the very nice property that there is a single global execution of the program's instructions that all threads agree on. This execution is just an interleaving of each thread's individual executions.

This does not hold if you start using the weaker atomic orderings.

The relative developer-friendliness of sequential consistency involves emitting memory fences, even on strongly-ordered platforms.

In practice, sequential consistency is rarely necessary for program correctness.

However sequential consistency is definitely the right choice if you're not confident about the other memory orders. Having your program run a bit slower than it needs to is certainly better than it running incorrectly!

It's also mechanically trivial to downgrade atomic operations to have a weaker consistency later on. Just change `SeqCst` to `Relaxed` and you're done! Of course, proving that this transformation is correct is a whole other matter.

## Acquire-Release

Acquire and Release are largely intended to be paired. They're perfectly suited for acquiring and releasing locks, and ensuring that critical sections don't overlap.

- An acquire access ensures that every access after it stays __after__ it. However operations that occur before an acquire are free to be reordered to occur after it.
- A release access ensures that every access before it stays __before__ it. However operations that occur after a release are free to be reordered to occur before it.

```
Thread A                Thread B

non-atomic write
relaxed atomic write
...
release x ----------------------┘
┌---------------------- acquire x
```

A simple spinlock:
```rust
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

fn main() {
    let lock = Arc::new(AtomicBool::new(false)); // value answers "am I locked?"

    // ... distribute lock to threads somehow ...

    // Try to acquire the lock by setting it to true
    while lock.compare_and_swap(false, true, Ordering::Acquire) { }
    // broke out of the loop, so we successfully acquired the lock!

    // ... scary data accesses ...

    // ok we're done, release the lock
    lock.store(false, Ordering::Release);
}
```

On strongly-ordered platforms most accesses have release or acquire semantics, making release and acquire often totally free.

## Relaxed

Relaxed accesses are the absolute weakest. They can be freely re-ordered and provide no happens-before relationship.

Relaxed operations are still atomic. That is, they don't count as data accesses and any read-modify-write operations done to them occur atomically.

For instance, incrementing a counter can be safely done by multiple threads using a relaxed `fetch_add` if you're not using the counter to synchronize any other accesses.

- Strongly-ordered platforms: Rarely a benefit, since they usually provide release-acquire semantics anyway.
- Weakly-ordered platforms: Cheaper.
