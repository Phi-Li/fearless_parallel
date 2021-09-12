use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::hint;
use std::thread;

fn main() {
    let spinlock = Arc::new(AtomicBool::new(false));

    // Distribute lock to threads
    let spinlock_clone = Arc::clone(&spinlock);
    let thread = thread::spawn(move || {
        // Try to acquire the lock by setting it to true
        while spinlock_clone.compare_exchange(
            false,
            true,
            Ordering::SeqCst,
            Ordering::SeqCst
        ).is_err() {
            hint::spin_loop();
        }
        // Broke out of while loop, so we successfully acquired the lock!

        // CRITICAL SECTION
        // ... scary data accesses ...

        // Ok we're done, release the lock
        spinlock_clone.store(false, Ordering::SeqCst)
    });

    // Wait for the other thread to release the lock
    while spinlock.load(Ordering::SeqCst) {
        hint::spin_loop();
    }

    if let Err(panic) = thread.join() {
        println!("Thread had an error: {:?}", panic);
    }
}
