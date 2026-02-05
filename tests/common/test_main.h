#pragma once
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace pomai::test
{

    using TestFn = void (*)();

    struct TestCase
    {
        const char *name;
        TestFn fn;
    };

    inline std::vector<TestCase> &Registry()
    {
        static std::vector<TestCase> r;
        return r;
    }

    struct Registrar
    {
        Registrar(const char *name, TestFn fn) { Registry().push_back(TestCase{name, fn}); }
    };

    template <typename A, typename B>
    inline bool ExpectEq(const A &a, const B &b)
    {
        if constexpr (std::is_integral_v<A> && std::is_integral_v<B> &&
                      !std::is_same_v<std::remove_cv_t<A>, bool> &&
                      !std::is_same_v<std::remove_cv_t<B>, bool>)
        {
            return std::cmp_equal(a, b);
        }
        else
        {
            return a == b;
        }
    }

    inline int RunAll()
    {
        int failed = 0;
        for (const auto &t : Registry())
        {
            try
            {
                t.fn();
            }
            catch (const std::exception &e)
            {
                std::cerr << "[FAILED] " << t.name << " exception: " << e.what() << "\n";
                ++failed;
            }
            catch (...)
            {
                std::cerr << "[FAILED] " << t.name << " unknown exception\n";
                ++failed;
            }
        }
        if (failed == 0)
        {
            // Keep quiet like bigtech default; uncomment if you want verbose pass logs.
            // std::cerr << "[OK] all tests passed\n";
        }
        return failed ? 2 : 0;
    }

} // namespace pomai::test

#define POMAI_EXPECT_TRUE(x)                             \
    do                                                   \
    {                                                    \
        if (!(x))                                        \
        {                                                \
            std::cerr << "EXPECT_TRUE failed: " #x "\n"; \
            std::abort();                                \
        }                                                \
    } while (0)

#define POMAI_EXPECT_EQ(a, b)                                    \
    do                                                           \
    {                                                            \
        auto _a = (a);                                           \
        auto _b = (b);                                           \
        if (!(::pomai::test::ExpectEq(_a, _b)))                  \
        {                                                        \
            std::cerr << "EXPECT_EQ failed: " #a " != " #b "\n"; \
            std::abort();                                        \
        }                                                        \
    } while (0)

#define POMAI_EXPECT_OK(st)                     \
    do                                          \
    {                                           \
        auto _st = (st);                        \
        if (!_st.ok())                          \
        {                                       \
            std::cerr << "EXPECT_OK failed: "   \
                      << _st.message() << "\n"; \
            std::abort();                       \
        }                                       \
    } while (0)

#define POMAI_TEST(name)                                      \
    static void name();                                       \
    static ::pomai::test::Registrar name##_reg(#name, &name); \
    static void name()
