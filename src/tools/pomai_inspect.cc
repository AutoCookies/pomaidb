#include "pomai/pomai.h"
#include "util/crc32c.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <cstring>
#include <iomanip>

namespace fs = std::filesystem;

// Minimal CLI tool for inspecting PomaiDB files

void PrintUsage() {
    std::cerr << "Usage:\n"
              << "  pomai_inspect checksum <file>\n"
              << "  pomai_inspect dump-manifest <manifest_file>\n";
}

int CmdChecksum(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Failed to open " << path << "\n";
        return 1;
    }
    
    in.seekg(0, std::ios::end);
    size_t sz = in.tellg();
    in.seekg(0, std::ios::beg);
    
    std::vector<char> buf(sz);
    in.read(buf.data(), sz);
    
    uint32_t crc = pomai::util::Crc32c(buf.data(), sz);
    std::cout << "CRC32C(" << path << ") = 0x" << std::hex << crc << std::dec << "\n";
    return 0;
}

int CmdDumpManifest(const std::string& path) {
    // Read raw file
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Failed to open " << path << "\n";
        return 1;
    }

    // Read all
    in.seekg(0, std::ios::end);
    size_t sz = in.tellg();
    in.seekg(0, std::ios::beg);
    
    if (sz < 4) {
        std::cerr << "File too small (<4 bytes)\n";
        return 1;
    }

    std::vector<char> buf(sz);
    in.read(buf.data(), sz);

    // Last 4 bytes is CRC
    size_t content_len = sz - 4;
    uint32_t stored_crc = 0;
    memcpy(&stored_crc, &buf[content_len], 4);
    
    uint32_t computed_crc = pomai::util::Crc32c(buf.data(), content_len);
    
    std::cout << "--- Manifest Dump ---\n";
    std::cout << "Path: " << path << "\n";
    std::cout << "Size: " << sz << " bytes\n";
    std::cout << "CRC Stored: 0x" << std::hex << stored_crc << "\n";
    std::cout << "CRC Computed: 0x" << computed_crc << "\n";
    
    if (stored_crc != computed_crc) {
        std::cout << "STATUS: CORRUPTED (CRC mismatch)\n";
    } else {
        std::cout << "STATUS: VALID\n";
    }
    
    std::cout << "\n[Content]\n";
    std::cout.write(buf.data(), content_len);
    std::cout << "\n---------------------\n";
    
    return (stored_crc == computed_crc) ? 0 : 1;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        PrintUsage();
        return 1;
    }

    std::string mode = argv[1];
    if (mode == "checksum") {
        if (argc != 3) { PrintUsage(); return 1; }
        return CmdChecksum(argv[2]);
    } else if (mode == "dump-manifest") {
        if (argc != 3) { PrintUsage(); return 1; }
        return CmdDumpManifest(argv[2]);
    } else {
        PrintUsage();
        return 1;
    }
}
