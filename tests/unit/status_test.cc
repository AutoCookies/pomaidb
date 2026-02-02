#include "pomai/status.h"
#include <gtest/gtest.h>

using namespace pomai;

TEST(StatusTest, DefaultIsOk)
{
    Status s;
    EXPECT_TRUE(s.ok());
    EXPECT_EQ(s.code(), ErrorCode::kOk);
    EXPECT_TRUE(s.message().empty());
}

TEST(StatusTest, OkFactory)
{
    Status s = Status::Ok();
    EXPECT_TRUE(s.ok());
    EXPECT_EQ(s.code(), ErrorCode::kOk);
}

TEST(StatusTest, InvalidArgument)
{
    Status s = Status::InvalidArgument("test message");
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code(), ErrorCode::kInvalidArgument);
    EXPECT_EQ(s.message(), "test message");
}

TEST(StatusTest, NotFound)
{
    Status s = Status::NotFound("resource not found");
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code(), ErrorCode::kNotFound);
    EXPECT_EQ(s.message(), "resource not found");
}

TEST(StatusTest, AlreadyExists)
{
    Status s = Status::AlreadyExists("duplicate key");
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code(), ErrorCode::kAlreadyExists);
    EXPECT_EQ(s.message(), "duplicate key");
}

TEST(StatusTest, PermissionDenied)
{
    Status s = Status::PermissionDenied("access denied");
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code(), ErrorCode::kPermissionDenied);
    EXPECT_EQ(s.message(), "access denied");
}

TEST(StatusTest, ResourceExhausted)
{
    Status s = Status::ResourceExhausted("out of memory");
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code(), ErrorCode::kResourceExhausted);
    EXPECT_EQ(s.message(), "out of memory");
}

TEST(StatusTest, FailedPrecondition)
{
    Status s = Status::FailedPrecondition("state invalid");
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code(), ErrorCode::kFailedPrecondition);
    EXPECT_EQ(s.message(), "state invalid");
}

TEST(StatusTest, Aborted)
{
    Status s = Status::Aborted("operation aborted");
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code(), ErrorCode::kAborted);
    EXPECT_EQ(s.message(), "operation aborted");
}

TEST(StatusTest, BusyAliasToAborted)
{
    Status s = Status::Busy("busy");
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code(), ErrorCode::kAborted);
    EXPECT_EQ(s.message(), "busy");
}

TEST(StatusTest, Corruption)
{
    Status s = Status::Corruption("data corrupted");
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code(), ErrorCode::kAborted);
    EXPECT_EQ(s.message(), "data corrupted");
}

TEST(StatusTest, IOError)
{
    Status s = Status::IOError("disk failure");
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code(), ErrorCode::kIO);
    EXPECT_EQ(s.message(), "disk failure");
}

TEST(StatusTest, IoErrorAlias)
{
    Status s = Status::IoError("io error");
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code(), ErrorCode::kIO);
    EXPECT_EQ(s.message(), "io error");
}

TEST(StatusTest, Internal)
{
    Status s = Status::Internal("internal error");
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code(), ErrorCode::kInternal);
    EXPECT_EQ(s.message(), "internal error");
}

TEST(StatusTest, Unknown)
{
    Status s = Status::Unknown("unknown error");
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code(), ErrorCode::kUnknown);
    EXPECT_EQ(s.message(), "unknown error");
}

TEST(StatusTest, CopyConstructor)
{
    Status s1 = Status::NotFound("original");
    Status s2 = s1;

    EXPECT_FALSE(s2.ok());
    EXPECT_EQ(s2.code(), ErrorCode::kNotFound);
    EXPECT_EQ(s2.message(), "original");
}

TEST(StatusTest, MoveConstructor)
{
    Status s1 = Status::InvalidArgument("move test");
    Status s2 = std::move(s1);

    EXPECT_FALSE(s2.ok());
    EXPECT_EQ(s2.code(), ErrorCode::kInvalidArgument);
    EXPECT_EQ(s2.message(), "move test");
}

TEST(StatusTest, EmptyMessage)
{
    Status s(ErrorCode::kInternal, "");
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.code(), ErrorCode::kInternal);
    EXPECT_TRUE(s.message().empty());
}

TEST(StatusTest, LongMessage)
{
    std::string long_msg(1000, 'x');
    Status s = Status::Internal(long_msg);
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.message().size(), 1000u);
    EXPECT_EQ(s.message(), long_msg);
}

TEST(StatusTest, PropagationPattern)
{
    // Test typical error propagation pattern
    auto fn1 = []() -> Status {
        return Status::IOError("low-level error");
    };

    auto fn2 = [&fn1]() -> Status {
        Status s = fn1();
        if (!s.ok()) return s;
        return Status::Ok();
    };

    Status result = fn2();
    EXPECT_FALSE(result.ok());
    EXPECT_EQ(result.code(), ErrorCode::kIO);
    EXPECT_EQ(result.message(), "low-level error");
}
