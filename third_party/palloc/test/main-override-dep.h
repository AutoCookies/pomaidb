#pragma once
// Issue #981: test overriding allocation in a DLL that is compiled independent of palloc. 
// This is imported by the `palloc-test-override` project.

#include <string>

class TestAllocInDll
{
public:
	__declspec(dllexport) std::string GetString();
};
