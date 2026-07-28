#pragma once

#include <vector>           // std::vector for the list of worker threads
#include <queue>            // std::queue for the task queue (FIFO)
#include <thread>           // std::thread -- actual OS threads
#include <mutex>            // std::mutex -- prevents threads from stepping on each other
#include <condition_variable> // std::condition_variable -- allows threads to wait/notify
#include <functional>       // std::function -- wraps any callable (lambda, function pointer)
#include <future>           // std::future -- gets results from async tasks
#include <atomic>           // std::atomic -- thread-safe counters without locks
#include <memory>           // std::unique_ptr for global singleton ownership

namespace clustering {

// ============================================================================
// ThreadPool -- A pool of worker threads that execute tasks in parallel.
//
// PROBLEM IT SOLVES:
//   Creating and destroying threads is EXPENSIVE (kernel calls, stack allocation).
//   Doing it for every small task would make parallel code SLOWER than single-threaded.
//
// SOLUTION:
//   Create a fixed number of worker threads ONCE at startup. They sit idle waiting
//   for tasks. When you enqueue a task, an idle worker picks it up and executes it.
//   When done, the worker goes back to waiting. No thread creation/destruction overhead.
//
// ANALOGY:
//   A thread pool is like a team of chefs in a restaurant kitchen.
//   Orders (tasks) come in through the ticket machine (queue). Any free chef
//   (worker thread) picks up the next ticket and cooks it.
//
// KEY CONCEPTS:
//   - Mutex (mutual exclusion lock): Like a "Do Not Disturb" sign. Only one thread
//     can hold the lock at a time. Used to protect shared data (the task queue).
//   - Condition variable: Lets threads SLEEP until there's work, then wakes them.
//     Without this, threads would spin in a loop wasting CPU (busy-waiting).
//   - Atomic: Thread-safe counter that doesn't need a mutex. Hardware guarantees
//     reads/writes are complete and visible to all threads.
//   - Future: A "promise" that a result will be available later. You can do other
//     work and then call future.get() to block until the result is ready.
//
// GLOBAL SINGLETON:
//   ThreadPool::global() returns a shared pool that's created once and reused.
//   This avoids creating multiple pools (which would oversubscribe CPU cores).
//   Cap: maximum 8 threads to prevent oversubscription.
// ============================================================================
class ThreadPool {
public:
    // Constructor: Create `threads` worker threads and start them.
    // If threads == 0: auto-detect hardware concurrency, cap at 8.
    // Workers immediately enter worker_loop() and wait for tasks.
    explicit ThreadPool(size_t threads = 0);

    // Destructor: Signal all threads to stop, wait for them to finish.
    // Sets stop_ flag, notifies all threads, then joins each one.
    ~ThreadPool();

    // Delete copy constructor and copy assignment.
    // You should NOT copy a thread pool (would create zombie threads).
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    // enqueue: Submit a task (any callable: function, lambda, functor) to the pool.
    // Returns: std::future<ReturnType> -- call .get() to retrieve the result.
    //
    // TEMPLATE: Works with any callable that returns any type.
    //   Example: auto fut = pool.enqueue([]{ return 42; });
    //            int result = fut.get();  // blocks until done, returns 42
    //
    // How it works:
    //   1. Wraps your function in a std::packaged_task (connects callable to future).
    //   2. Wraps that in a shared_ptr (needed because std::function requires copyable).
    //   3. Pushes the task into the queue (under mutex lock).
    //   4. Increments active_tasks_ counter.
    //   5. Wakes ONE sleeping worker thread via condition_.notify_one().
    //   6. Returns the future immediately (your code continues).
    //
    // Thread safety: The push is guarded by queue_mutex_. The worker will pop
    //   and execute the task later, possibly on a different core.
    template<class F>
    auto enqueue(F&& f) -> std::future<decltype(f())>;

    // wait_all: Block until ALL tasks in the queue are completed.
    // Useful for synchronization: "finish all current work before continuing."
    // Waits until tasks_ is empty AND active_tasks_ == 0.
    void wait_all();

    // size: How many worker threads are in this pool?
    size_t size() const { return workers_.size(); }

    // global: Get the singleton shared thread pool.
    // Creates it on first call, returns existing one on subsequent calls.
    static ThreadPool& global();

private:
    // worker_loop: The function EACH worker thread runs in an infinite loop.
    //   1. Wait for a task to appear in the queue (sleeps on condition variable).
    //   2. Pop a task from the front of the queue.
    //   3. Execute the task.
    //   4. Decrement active_tasks_ and notify wait_all() if queue is empty.
    //   5. Go back to step 1.
    //
    //   Exits the loop ONLY when stop_ is true AND the queue is empty.
    void worker_loop();

    std::vector<std::thread> workers_;             // The actual OS threads
    std::queue<std::function<void()>> tasks_;      // FIFO task queue
                                                   // std::function<void()> wraps any callable

    std::mutex queue_mutex_;                       // Protects the task queue and counters
    std::condition_variable condition_;            // "Hey workers, there's a new task!"
    std::condition_variable finished_;             // "All tasks done!" (for wait_all)
    std::atomic<bool> stop_;                       // Shutdown flag (atomic, no mutex needed)
    std::atomic<size_t> active_tasks_;             // How many tasks are running or queued

    static std::unique_ptr<ThreadPool> global_instance_; // The singleton pool
};

// ============================================================================
// enqueue() TEMPLATE IMPLEMENTATION
//
// This must be in the header file because it's a template. The compiler needs
// to see the full implementation to generate code for each callable type.
//
// Breaking it down step by step:
//
// 1. using return_type = decltype(f());  -- Figure out what type f() returns.
//    decltype deduces the return type without actually calling the function.
//
// 2. auto task = std::make_shared<std::packaged_task<return_type()>>(f);
//    packaged_task wraps our function so it can be called later AND gives us
//    a future to get the result. shared_ptr is needed because std::function
//    requires its contents to be copyable (packaged_task is move-only).
//
// 3. std::future<return_type> result = task->get_future();
//    Get the future BEFORE enqueueing. The future is the read-end of a
//    one-shot channel: set_value() on the task end, get() on the future end.
//
// 4. { lock; tasks_.emplace([task]() { (*task)(); }); ++active_tasks_; }
//    Push a lambda that calls the packaged_task. The lambda captures `task`
//    by value (increments the shared_ptr ref count). The lock_guard ensures
//    only one thread modifies the queue at a time. RAII: lock auto-releases
//    when leaving the {} scope.
//
// 5. condition_.notify_one();  -- Wake ONE sleeping worker.
//    Using notify_one (not notify_all) because only ONE worker can take
//    the next task from the queue anyway (mutex serializes access).
//
// 6. return result;  -- Return the future. The caller can later call .get()
//    to retrieve the return value. If the task hasn't finished yet, .get()
//    blocks until it's done.
// ============================================================================
template<class F>
auto ThreadPool::enqueue(F&& f) -> std::future<decltype(f())> {
    using return_type = decltype(f());

    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::forward<F>(f)
    );

    std::future<return_type> result = task->get_future();
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (stop_) {
            throw std::runtime_error("enqueue on stopped ThreadPool");
        }
        tasks_.emplace([task]() { (*task)(); });
        ++active_tasks_;
    }
    condition_.notify_one();
    return result;
}

} // namespace clustering
