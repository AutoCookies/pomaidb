// Unit tests for VFS: PosixEnv (via Env::Default()) and InMemoryEnv.

#include "tests/common/test_main.h"
#include "tests/common/test_tmpdir.h"
#include "pomai/env.h"
#include <cstring>
#include <string>

namespace pomai {

// ---- PosixEnv (Env::Default()) ----
POMAI_TEST(Env_Default_NotNull) {
  Env* env = Env::Default();
  POMAI_EXPECT_TRUE(env != nullptr);
}

POMAI_TEST(Env_Posix_CreateDirAndFile) {
  std::string dir = test::TempDir("env_posix");
  Env* env = Env::Default();
  POMAI_EXPECT_OK(env->CreateDirIfMissing(dir));
  std::string path = dir + "/foo.txt";
  std::unique_ptr<WritableFile> wf;
  POMAI_EXPECT_OK(env->NewWritableFile(path, &wf));
  POMAI_EXPECT_TRUE(wf != nullptr);
  const char* msg = "hello world";
  POMAI_EXPECT_OK(wf->Append(Slice(msg, std::strlen(msg))));
  POMAI_EXPECT_OK(wf->Sync());
  POMAI_EXPECT_OK(wf->Close());
  uint64_t size = 0;
  POMAI_EXPECT_OK(env->GetFileSize(path, &size));
  POMAI_EXPECT_EQ(size, static_cast<uint64_t>(std::strlen(msg)));
  POMAI_EXPECT_OK(env->FileExists(path));
}

POMAI_TEST(Env_Posix_ReadBack) {
  std::string dir = test::TempDir("env_posix_read");
  Env* env = Env::Default();
  POMAI_EXPECT_OK(env->CreateDirIfMissing(dir));
  std::string path = dir + "/bar.dat";
  {
    std::unique_ptr<WritableFile> wf;
    POMAI_EXPECT_OK(env->NewWritableFile(path, &wf));
    const char* data = "abc";
    POMAI_EXPECT_OK(wf->Append(Slice(data, 3)));
    POMAI_EXPECT_OK(wf->Close());
  }
  std::unique_ptr<RandomAccessFile> raf;
  POMAI_EXPECT_OK(env->NewRandomAccessFile(path, &raf));
  Slice s;
  POMAI_EXPECT_OK(raf->Read(0, 3, &s));
  POMAI_EXPECT_EQ(s.size(), static_cast<size_t>(3));
  POMAI_EXPECT_TRUE(std::memcmp(s.data(), "abc", 3) == 0);
  POMAI_EXPECT_OK(raf->Close());
}

// ---- InMemoryEnv ----
POMAI_TEST(Env_InMemory_WriteAndRead) {
  std::unique_ptr<Env> env = Env::NewInMemory();
  POMAI_EXPECT_TRUE(env != nullptr);
  std::unique_ptr<WritableFile> wf;
  POMAI_EXPECT_OK(env->NewWritableFile("/f", &wf));
  POMAI_EXPECT_OK(wf->Append(Slice("xyz", 3)));
  POMAI_EXPECT_OK(wf->Close());
  POMAI_EXPECT_OK(env->FileExists("/f"));
  uint64_t size = 0;
  POMAI_EXPECT_OK(env->GetFileSize("/f", &size));
  POMAI_EXPECT_EQ(size, 3u);
  std::unique_ptr<RandomAccessFile> raf;
  POMAI_EXPECT_OK(env->NewRandomAccessFile("/f", &raf));
  Slice s;
  POMAI_EXPECT_OK(raf->Read(0, 3, &s));
  POMAI_EXPECT_EQ(s.size(), 3u);
  POMAI_EXPECT_TRUE(std::memcmp(s.data(), "xyz", 3) == 0);
}

POMAI_TEST(Env_InMemory_AppendableFile) {
  std::unique_ptr<Env> env = Env::NewInMemory();
  std::unique_ptr<WritableFile> wf;
  POMAI_EXPECT_OK(env->NewWritableFile("/a", &wf));
  POMAI_EXPECT_OK(wf->Append(Slice("one", 3)));
  POMAI_EXPECT_OK(wf->Close());
  POMAI_EXPECT_OK(env->NewAppendableFile("/a", &wf));
  POMAI_EXPECT_OK(wf->Append(Slice("two", 3)));
  POMAI_EXPECT_OK(wf->Close());
  uint64_t size = 0;
  POMAI_EXPECT_OK(env->GetFileSize("/a", &size));
  POMAI_EXPECT_EQ(size, 6u);
  std::unique_ptr<RandomAccessFile> raf;
  POMAI_EXPECT_OK(env->NewRandomAccessFile("/a", &raf));
  Slice s;
  POMAI_EXPECT_OK(raf->Read(0, 6, &s));
  POMAI_EXPECT_TRUE(s.size() == 6 && std::memcmp(s.data(), "onetwo", 6) == 0);
}

POMAI_TEST(Env_InMemory_FileMapping) {
  std::unique_ptr<Env> env = Env::NewInMemory();
  std::unique_ptr<WritableFile> wf;
  POMAI_EXPECT_OK(env->NewWritableFile("/m", &wf));
  POMAI_EXPECT_OK(wf->Append(Slice("mapped", 6)));
  POMAI_EXPECT_OK(wf->Close());
  std::unique_ptr<FileMapping> map;
  POMAI_EXPECT_OK(env->NewFileMapping("/m", &map));
  POMAI_EXPECT_TRUE(map != nullptr);
  POMAI_EXPECT_EQ(map->Size(), 6u);
  POMAI_EXPECT_TRUE(map->Data() != nullptr);
  POMAI_EXPECT_TRUE(std::memcmp(map->Data(), "mapped", 6) == 0);
}

POMAI_TEST(Env_InMemory_RenameAndDelete) {
  std::unique_ptr<Env> env = Env::NewInMemory();
  std::unique_ptr<WritableFile> wf;
  POMAI_EXPECT_OK(env->NewWritableFile("/old", &wf));
  POMAI_EXPECT_OK(wf->Close());
  POMAI_EXPECT_OK(env->RenameFile("/old", "/new"));
  POMAI_EXPECT_TRUE(!env->FileExists("/old").ok());
  POMAI_EXPECT_OK(env->FileExists("/new"));
  POMAI_EXPECT_OK(env->DeleteFile("/new"));
  POMAI_EXPECT_TRUE(!env->FileExists("/new").ok());
}

}  // namespace pomai
