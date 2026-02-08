#include "pomai/pomai.h"
#include "storage/manifest/manifest.h"
#include "pomai/iterator.h"
#include "util/crc32c.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <cstring>
#include <iomanip>
#include <sstream>

namespace fs = std::filesystem;

// Minimal CLI tool for inspecting PomaiDB files

void PrintUsage() {
    std::cerr << "Usage:\n"
              << "  pomai_inspect info <db_path>                                     - Show database summary\n"
              << "  pomai_inspect membranes <db_path>                                - List all membranes\n"
              << "  pomai_inspect wal-stats <db_path>                                - Show WAL statistics\n"
              << "  pomai_inspect segment-stats <db_path>                            - Show segment statistics\n"
              << "  pomai_inspect scan <db_path> [--membrane=<name>] [--format=...]  - Export vectors\n"
              << "  pomai_inspect checksum <file>                                    - Compute CRC32C\n"
              << "  pomai_inspect dump-manifest <manifest_file>                      - Dump manifest content\n";
}

int CmdInfo(const std::vector<std::string>& args) {
    if (args.empty()) {
        std::cerr << "Error: info requires <db_path>\n";
        PrintUsage();
        return 1;
    }
    
    std::string db_path = args[0];
    
    // Open database
    pomai::DBOptions opt;
    opt.path = db_path;
    
    std::unique_ptr<pomai::DB> db;
    auto st = pomai::DB::Open(opt, &db);
    if (!st.ok()) {
        std::cerr << "Error opening database: " << st.message() << "\n";
        return 1;
    }
    
    // Get basic info
    std::cout << "Database Info\n";
    std::cout << "=============\n";
    std::cout << "Path: " << db_path << "\n";
    
    // Check if path exists and get size
    if (fs::exists(db_path)) {
        size_t total_size = 0;
        for (const auto& entry : fs::recursive_directory_iterator(db_path)) {
            if (entry.is_regular_file()) {
                total_size += entry.file_size();
            }
        }
        std::cout << "Size: " << (total_size / (1024 * 1024)) << " MB\n";
    }
    
    // List membranes
    std::vector<std::string> membranes;
    st = db->ListMembranes(&membranes);
    if (st.ok()) {
        std::cout << "Membranes: " << membranes.size() << "\n";
        for (const auto& m : membranes) {
            std::cout << "  - " << m << "\n";
        }
    }
    
    db->Close();
    std::cout << "\n";
    return 0;
}

int CmdMembranes(const std::vector<std::string>& args) {
    if (args.empty()) {
        std::cerr << "Error: membranes requires <db_path>\n";
        PrintUsage();
        return 1;
    }
    
    std::string db_path = args[0];
    
    // Open database
    pomai::DBOptions opt;
    opt.path = db_path;
    
    std::unique_ptr<pomai::DB> db;
    auto st = pomai::DB::Open(opt, &db);
    if (!st.ok()) {
        std::cerr << "Error opening database: " << st.message() << "\n";
        return 1;
    }
    
    // List membranes
    std::vector<std::string> membranes;
    st = db->ListMembranes(&membranes);
    if (!st.ok()) {
        std::cerr << "Error listing membranes: " << st.message() << "\n";
        db->Close();
        return 1;
    }
    
    std::cout << "Membranes (" << membranes.size() << "):\n";
    std::cout << "===================\n";
    
    for (const auto& membrane_name : membranes) {
        std::cout << "\nMembrane: " << membrane_name << "\n";

        pomai::MembraneSpec spec;
        auto spec_st = pomai::storage::Manifest::GetMembrane(opt.path, membrane_name, &spec);
        if (spec_st.ok()) {
            std::cout << "  Kind: " << (spec.kind == pomai::MembraneKind::kRag ? "RAG" : "VECTOR") << "\n";
        }

        if (!spec_st.ok() || spec.kind == pomai::MembraneKind::kVector) {
            // Try to get iterator to count vectors
            std::unique_ptr<pomai::SnapshotIterator> it;
            st = db->NewIterator(membrane_name, &it);
            if (st.ok()) {
                size_t count = 0;
                while (it->Valid()) {
                    count++;
                    it->Next();
                }
                std::cout << "  Vectors: " << count << "\n";
            }
        } else {
            std::cout << "  Chunks: (use rag tooling)\n";
        }
    }
    
    std::cout << "\n";
    db->Close();
    return 0;
}

int CmdWalStats(const std::vector<std::string>& args) {
    if (args.empty()) {
        std::cerr << "Error: wal-stats requires <db_path>\n";
        PrintUsage();
        return 1;
    }
    
    std::string db_path = args[0];
    
    if (!fs::exists(db_path)) {
        std::cerr << "Error: Path does not exist: " << db_path << "\n";
        return 1;
    }
    
    std::cout << "WAL Statistics\n";
    std::cout << "==============\n";
    
    size_t total_wal_size = 0;
    int wal_file_count = 0;
    
    // Scan for membranes
    for (const auto& membrane_entry : fs::directory_iterator(db_path)) {
        if (!membrane_entry.is_directory()) continue;
        
        std::string membrane_name = membrane_entry.path().filename().string();
        if (membrane_name == "." || membrane_name == "..") continue;
        
        // Check each shard
        for (const auto& shard_entry : fs::directory_iterator(membrane_entry.path())) {
            if (!shard_entry.is_directory()) continue;
            if (!shard_entry.path().filename().string().starts_with("shard_")) continue;
            
            // Find WAL files in this shard
            auto wal_path = shard_entry.path() / "wal";
            if (fs::exists(wal_path) && fs::is_regular_file(wal_path)) {
                size_t size = fs::file_size(wal_path);
                total_wal_size += size;
                wal_file_count++;
                
                std::cout << "\n  " << membrane_name << " / " 
                          << shard_entry.path().filename().string() << ":\n";
                std::cout << "    WAL: " << (size / 1024) << " KB\n";
            }
        }
    }
    
    std::cout << "\nTotal:\n";
    std::cout << "  Files: " << wal_file_count << "\n";
    std::cout << "  Size: " << (total_wal_size / (1024 * 1024)) << " MB\n\n";
    
    return 0;
}

int CmdSegmentStats(const std::vector<std::string>& args) {
    if (args.empty()) {
        std::cerr << "Error: segment-stats requires <db_path>\n";
        PrintUsage();
        return 1;
    }
    
    std::string db_path = args[0];
    
    if (!fs::exists(db_path)) {
        std::cerr << "Error: Path does not exist: " << db_path << "\n";
        return 1;
    }
    
    std::cout << "Segment Statistics\n";
    std::cout << "==================\n";
    
    size_t total_segment_size = 0;
    int segment_file_count = 0;
    
    // Scan for membranes
    for (const auto& membrane_entry : fs::directory_iterator(db_path)) {
        if (!membrane_entry.is_directory()) continue;
        
        std::string membrane_name = membrane_entry.path().filename().string();
        if (membrane_name == "." || membrane_name == "..") continue;
        
        std::cout << "\nMembrane: " << membrane_name << "\n";
        
        // Check each shard
        for (const auto& shard_entry : fs::directory_iterator(membrane_entry.path())) {
            if (!shard_entry.is_directory()) continue;
            if (!shard_entry.path().filename().string().starts_with("shard_")) continue;
            
            std::string shard_name = shard_entry.path().filename().string();
            
            // Find segment files in this shard
            std::vector<std::string> segments;
            for (const auto& file : fs::directory_iterator(shard_entry.path())) {
                if (file.is_regular_file()) {
                    std::string fname = file.path().filename().string();
                    if (fname.starts_with("seg_") && fname.ends_with(".dat")) {
                        segments.push_back(fname);
                        segment_file_count++;
                        total_segment_size += fs::file_size(file.path());
                    }
                }
            }
            
            if (!segments.empty()) {
                std::cout << "  " << shard_name << ": " << segments.size() << " segments\n";
                for (const auto& seg : segments) {
                    auto seg_path = shard_entry.path() / seg;
                    size_t size = fs::file_size(seg_path);
                    std::cout << "    " << seg << ": " << (size / 1024) << " KB\n";
                }
            }
        }
    }
    
    std::cout << "\nTotal:\n";
    std::cout << "  Segments: " << segment_file_count << "\n";
    std::cout << "  Size: " << (total_segment_size / (1024 * 1024)) << " MB\n\n";
    
    return 0;
}

int CmdScan(const std::vector<std::string>& args) {
    if (args.empty()) {
        std::cerr << "Error: scan requires <db_path>\n";
        PrintUsage();
        return 1;
    }
    
    std::string db_path = args[0];
    std::string membrane = "__default__";
    std::string format = "json";
    
    // Parse optional arguments
    for (size_t i = 1; i < args.size(); ++i) {
        std::string arg = args[i];
        if (arg.find("--membrane=") == 0) {
            membrane = arg.substr(11);
        } else if (arg.find("--format=") == 0) {
            format = arg.substr(9);
            if (format != "json" && format != "binary") {
                std::cerr << "Error: format must be 'json' or 'binary'\n";
                return 1;
            }
        } else {
            std::cerr << "Error: unknown argument '" << arg << "'\n";
            PrintUsage();
            return 1;
        }
    }
    
    // Open database - we need to read membrane metadata first
    pomai::DBOptions opt;
    opt.path = db_path;
    
    // Read membrane metadata from manifest to get dim/shard_count
    pomai::MembraneSpec mspec;
    std::unique_ptr<pomai::DB> db;
    auto st = pomai::DB::Open(opt, &db);
    
    // If open fails, try to infer from default membrane
    // Most DBs will have __default__ membrane with correct settings
    if (!st.ok()) {
        std::cerr << "Error opening database: " << st.message() << "\n";
        std::cerr << "Hint: Database may not be initialized or path is incorrect\n";
        return 1;
    }
    
    // Get iterator
    std::unique_ptr<pomai::SnapshotIterator> it;
    st = db->NewIterator(membrane, &it);
    if (!st.ok()) {
        std::cerr << "Error creating iterator for membrane '" << membrane << "': " << st.message() << "\n";
        db->Close();
        return 1;
    }
    
    // Export data
    if (format == "json") {
        std::cout << "[\n";
        bool first = true;
        size_t count = 0;
        
        while (it->Valid()) {
            if (!first) {
                std::cout << ",\n";
            }
            first = false;
            
            auto id = it->id();
            auto vec = it->vector();
            
            std::cout << "  {\"id\": " << id << ", \"vector\": [";
            for (size_t i = 0; i < vec.size(); ++i) {
                if (i > 0) std::cout << ", ";
                std::cout << vec[i];
            }
            std::cout << "]}";
            
            count++;
            it->Next();
        }
        
        std::cout << "\n]\n";
        std::cerr << "Exported " << count << " vectors from membrane '" << membrane << "'\n";
    } else {
        // Binary format: [count:8][id:8,dim:4,vec:dim*4]*count
        std::vector<std::pair<uint64_t, std::vector<float>>> entries;
        
        while (it->Valid()) {
            auto id = it->id();
            auto vec = it->vector();
            entries.emplace_back(id, std::vector<float>(vec.begin(), vec.end()));
            it->Next();
        }
        
        // Write to stdout
        uint64_t count = entries.size();
        std::cout.write(reinterpret_cast<const char*>(&count), sizeof(count));
        
        for (const auto& [id, vec] : entries) {
            uint64_t vid = id;
            uint32_t dim = static_cast<uint32_t>(vec.size());
            std::cout.write(reinterpret_cast<const char*>(&vid), sizeof(vid));
            std::cout.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
            std::cout.write(reinterpret_cast<const char*>(vec.data()), dim * sizeof(float));
        }
        
        std::cerr << "Exported " << count << " vectors from membrane '" << membrane << "' (binary)\n";
    }
    
    db->Close();
    return 0;
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
    if (mode == "info") {
        std::vector<std::string> args(argv + 2, argv + argc);
        return CmdInfo(args);
    } else if (mode == "membranes") {
        std::vector<std::string> args(argv + 2, argv + argc);
        return CmdMembranes(args);
    } else if (mode == "wal-stats") {
        std::vector<std::string> args(argv + 2, argv + argc);
        return CmdWalStats(args);
    } else if (mode == "segment-stats") {
        std::vector<std::string> args(argv + 2, argv + argc);
        return CmdSegmentStats(args);
    } else if (mode == "scan") {
        std::vector<std::string> args(argv + 2, argv + argc);
        return CmdScan(args);
    } else if (mode == "checksum") {
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
