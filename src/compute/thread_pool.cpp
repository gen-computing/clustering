// ============================================================================
// ThreadPool implementation -- Worker thread lifecycle management.
//
// The ThreadPool creates N worker threads at construction and keeps them alive
// until destruction. Workers sleep on a condition variable when there's no work,
// consuming zero CPU. When a task is enqueued, one worker wakes up, grabs the
// task from the queue, executes it, then goes back to sleep.
//
// DESIGN CHOICES:
//   - Fixed number of threads (no dynamic scaling) for predictable performance.
//   - Cap at 8 threads: beyond this, contention and context switching hurt more
//     than they help for our workload type (CPU-bound, not I/O-bound).
//   - Global singleton: avoids creating multiple pools that would oversubscribe
//     the CPU (e.g., 4 pools of 4 threads on a 4-core machine = 16 threads).
// ============================================================================

#include "clustering/thread_pool.h"
#include <stdexcept>   // std::runtime_error
#include <mutex>       // std::call_once, std::once_flag

namespace clustering {

// The global singleton pool. Created lazily (first call to global()).
// std::unique_ptr ensures it's properly deleted when the program exits.
std::unique_ptr<ThreadPool> ThreadPool::global_instance_ = nullptr;
static std::once_flag global_pool_flag;

// ============================================================================
// CONSTRUCTOR -- CREATE AND START WORKER THREADS
//
// 1. Determine thread count: auto-detect or use user-specified value.
// 2. Create each worker thread, which immediately enters worker_loop().
// 3. Worker threads start running RIGHT AWAY (they'll find empty queue and sleep).
// ============================================================================

ThreadPool::ThreadPool(size_t threads)
    : stop_(false), active_tasks_(0)
{
    // Auto-detect: use the number of hardware threads available.
    if (threads == 0) {
        // std::thread::hardware_concurrency() returns the number of logical
        // cores (including hyperthreading). On a 4-core/8-thread CPU, returns 8.
        threads = std::max(size_t(1), size_t(std::thread::hardware_concurrency()));

        // CAP at 8 to prevent oversubscription on high-core-count machines.
        // More threads than physical cores causes context switching overhead
        // that makes things SLOWER, not faster (especially for CPU-bound work).
        threads = std::min(threads, size_t(8));
    }

    // Reserve space in the vector to avoid reallocation during thread creation.
    // Reserving means we can emplace_back without moving existing entries.
    workers_.reserve(threads);

    // Create and start each worker thread.
    // emplace_back constructs the std::thread IN PLACE (no copy/move).
    // Each thread starts executing worker_loop() on `this` (the pool instance).
    for (size_t i = 0; i < threads; ++i) {
        workers_.emplace_back(&ThreadPool::worker_loop, this);
    }
}

// ============================================================================
// DESTRUCTOR -- GRACEFUL SHUTDOWN
//
// 1. Set stop_ flag to tell workers "no more tasks, finish and exit."
// 2. Wake ALL sleeping workers (they'll see stop_ and exit their loop).
// 3. Join each thread (wait for it to finish). This blocks until all workers exit.
//
// IMPORTANT: stop_ must be set under the mutex and notification sent
// under the same mutex, otherwise a worker could miss the wake-up
// (the notorious "lost wakeup" race condition).
// ============================================================================

ThreadPool::~ThreadPool() {
    {
        // Lock the mutex before modifying stop_.
        // This ensures the memory write to stop_ is visible to all threads
        // (mutex unlock provides a memory barrier).
        std::lock_guard<std::mutex> lock(queue_mutex_);
        stop_ = true;
    }
    // Notify ALL workers. Each will wake up, check stop_, and exit.
    // notify_all (not notify_one) because ALL workers need to shut down.
    condition_.notify_all();

    // Wait for each worker thread to finish.
    // join() blocks until the thread's function returns.
    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

// ============================================================================
// worker_loop() -- WHAT EACH WORKER THREAD RUNS
//
// This is an INFINITE LOOP. Each worker:
//   1. Acquires the queue mutex.
//   2. Waits on the condition variable (sleeps until notified).
//   3. The condition_variable::wait predicate checks:
//        - Is stop_ true? (time to shut down)
//        - Is there a task in the queue? (work to do)
//      If NEITHER is true, the worker goes back to sleep.
//      This avoids "spurious wakeups" (OS waking the thread for no reason).
//   4. If stop_ and no tasks: EXIT the loop (thread terminates).
//   5. Otherwise: pop a task from the front of the queue.
//   6. Release the lock (unique_lock destructor) so other workers can proceed.
//   7. Execute the task.
//   8. Lock again, decrement active_tasks_, notify wait_all() listeners.
//   9. Go back to step 1.
// ============================================================================

void ThreadPool::worker_loop() {
    while (true) {
        std::function<void()> task;  // The task to execute (initially empty)

        {
            // Acquire a UNIQUE lock (needed for condition_variable::wait).
            // std::unique_lock can be locked/unlocked; lock_guard cannot.
            std::unique_lock<std::mutex> lock(queue_mutex_);

            // WAIT until there's work to do or we should stop.
            // The predicate lambda is called each time the thread wakes up.
            // It returns true when the thread should PROCEED (grab a task).
            //
            // condition_.wait(lock, predicate) is equivalent to:
            //   while (!predicate()) { condition_.wait(lock); }
            // This handles spurious wakeups automatically.
            condition_.wait(lock, [this] {
                return stop_ || !tasks_.empty();
            });

            // If we were told to stop AND the queue is empty, exit the thread.
            // We drain the queue even during shutdown to avoid losing work.
            if (stop_ && tasks_.empty()) {
                return;  // Thread exits here
            }

            // Move the task out of the queue.
            // std::move avoids copying the std::function (which could be expensive
            // if the function captures large objects).
            task = std::move(tasks_.front());
            tasks_.pop();
        }
        // Lock is released here (unique_lock destructor).

        // EXECUTE the task outside the locked region.
        // This is CRUCIAL: if we held the lock during execution, only one
        // thread could work at a time, defeating the purpose of a thread pool.
        task();

        {
            // Lock briefly to update the counter and notify waiters.
            std::lock_guard<std::mutex> lock(queue_mutex_);
            --active_tasks_;
        }
        // Notify wait_all() that a task has completed.
        // If this was the last active task, wait_all() will unblock.
        finished_.notify_one();
    }
}

// ============================================================================
// wait_all() -- BLOCK UNTIL ALL QUEUED TASKS COMPLETE
//
// Used when you need a synchronization point: "finish all current work
// before continuing." The predicate checks:
//   - tasks_ empty: no more work waiting to be picked up
//   - active_tasks_ == 0: no worker is currently executing a task
//
// This blocks the CALLING thread (not the workers) until conditions are met.
// ============================================================================

void ThreadPool::wait_all() {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    finished_.wait(lock, [this] {
        return tasks_.empty() && active_tasks_ == 0;
    });
}

// ============================================================================
// global() -- GLOBAL SINGLETON ACCESS
//
// Creates the pool on first call (lazy initialization).
// Returns a reference to the same pool on every subsequent call.
//
// Thread-safe: if two threads call global() simultaneously,
// only one pool is created (but our current usage is single-threaded
// for pool creation).
// ============================================================================

ThreadPool& ThreadPool::global() {
    std::call_once(global_pool_flag, []() {
        global_instance_ = std::make_unique<ThreadPool>();
    });
    return *global_instance_;
}

} // namespace clustering
