#include "tests/common/test_main.h"
#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

#include "core/shard/mailbox.h"

POMAI_TEST(Mailbox_BasicMpsc)
{
  using Q = pomai::core::BoundedMpscQueue<std::uint64_t>;
  Q q(/*cap*/ 1024);

  std::atomic<std::uint64_t> sum{0};

  std::jthread consumer([&]
                        {
    for (;;) {
      auto v = q.PopBlocking();
      if (!v.has_value()) break;
      sum.fetch_add(*v, std::memory_order_relaxed);
    } });

  constexpr int kProducers = 4;
  constexpr int kPer = 2000;

  std::vector<std::jthread> prod;
  for (int p = 0; p < kProducers; ++p)
  {
    prod.emplace_back([&]
                      {
      for (int i = 1; i <= kPer; ++i) {
        POMAI_EXPECT_TRUE(q.PushBlocking(static_cast<std::uint64_t>(i)));
      } });
  }
  prod.clear();

  q.Close();
  consumer.join();

  // Each producer pushes 1..kPer
  const std::uint64_t expected_one = (static_cast<std::uint64_t>(kPer) * (kPer + 1)) / 2;
  const std::uint64_t expected = expected_one * kProducers;
  POMAI_EXPECT_EQ(sum.load(), expected);
  POMAI_EXPECT_EQ(q.Size(), 0);
}
