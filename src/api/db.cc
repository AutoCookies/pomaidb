#include "pomai/pomai.h"
#include <memory>
#include <vector>
#include "core/engine/engine.h"

namespace pomai
{

    class DbImpl final : public DB
    {
    public:
        explicit DbImpl(DBOptions opt) : eng_(std::move(opt)) {}

        Status Open() { return eng_.Open(); }

        Status Put(VectorId id, std::span<const float> vec) override { return eng_.Put(id, vec); }
        Status Delete(VectorId id) override { return eng_.Delete(id); }

        Status Search(std::span<const float> query, std::uint32_t topk, SearchResult *out) override
        {
            hits_.clear();
            auto st = eng_.Search(query, topk, &hits_);
            if (!st.ok())
                return st;
            out->hits = std::span<const SearchHit>{hits_.data(), hits_.size()};
            return Status::Ok();
        }

        Status Flush() override { return eng_.Flush(); }
        Status Close() override { return eng_.Close(); }

    private:
        core::Engine eng_;
        std::vector<SearchHit> hits_;
    };

    Status DB::Open(const DBOptions &options, std::unique_ptr<DB> *out)
    {
        if (!out)
            return Status::InvalidArgument("out=null");
        auto db = std::make_unique<DbImpl>(options);
        auto st = db->Open();
        if (!st.ok())
            return st;
        *out = std::move(db);
        return Status::Ok();
    }

} // namespace pomai
