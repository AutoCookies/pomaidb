#pragma once

#include <pomai/core/types.h>

namespace pomai
{
    inline SearchRequest NormalizeSearchRequest(const SearchRequest &req)
    {
        SearchRequest out = req;
        out.max_rerank_k = NormalizeMaxRerankK(out);
        out.candidate_k = NormalizeCandidateK(out);
        out.graph_ef = NormalizeGraphEf(out, out.candidate_k);
        return out;
    }
} // namespace pomai
