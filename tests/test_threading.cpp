#include <gtest/gtest.h>
#include "clustering/thread_pool.h"
#include <atomic>
#include <vector>

using namespace clustering;

TEST(ThreadPoolTest, BasicEnqueue) {
    ThreadPool pool(2);

    auto result = pool.enqueue([]() {
        return 42;
    });

    EXPECT_EQ(result.get(), 42);
}

TEST(ThreadPoolTest, MultipleTasks) {
    ThreadPool pool(4);
    std::atomic<int> counter(0);

    std::vector<std::future<void>> futures;
    for (int i = 0; i < 100; ++i) {
        futures.push_back(pool.enqueue([&counter]() {
            counter++;
        }));
    }

    for (auto& f : futures) {
        f.get();
    }

    EXPECT_EQ(counter.load(), 100);
}

TEST(ThreadPoolTest, WaitAll) {
    ThreadPool pool(4);
    std::atomic<int> counter(0);

    for (int i = 0; i < 50; ++i) {
        pool.enqueue([&counter]() {
            counter++;
        });
    }

    pool.wait_all();
    EXPECT_EQ(counter.load(), 50);
}

TEST(ThreadPoolTest, LargeNumberOfTasks) {
    ThreadPool pool(8);
    std::atomic<long long> sum(0);

    std::vector<std::future<void>> futures;
    for (long long i = 0; i < 10000; ++i) {
        futures.push_back(pool.enqueue([&sum, i]() {
            sum += i;
        }));
    }

    for (auto& f : futures) {
        f.get();
    }

    long long expected = 10000LL * 9999LL / 2;
    EXPECT_EQ(sum.load(), expected);
}

TEST(ThreadPoolTest, TaskThatReturnsValue) {
    ThreadPool pool(2);

    auto result = pool.enqueue([]() {
        return std::string("hello");
    });

    EXPECT_EQ(result.get(), "hello");
}

TEST(ThreadPoolTest, TaskThatThrows) {
    ThreadPool pool(2);

    auto result = pool.enqueue([]() -> int {
        throw std::runtime_error("test error");
    });

    EXPECT_THROW(result.get(), std::runtime_error);
}

TEST(ThreadPoolTest, ConcurrentIncrement) {
    ThreadPool pool(8);
    int counter = 0;
    std::mutex mtx;

    std::vector<std::future<void>> futures;
    for (int i = 0; i < 1000; ++i) {
        futures.push_back(pool.enqueue([&counter, &mtx]() {
            std::lock_guard<std::mutex> lock(mtx);
            counter++;
        }));
    }

    for (auto& f : futures) {
        f.get();
    }

    EXPECT_EQ(counter, 1000);
}
