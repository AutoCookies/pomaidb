struct PendingInsert
{
    uint64_t label;
    // We store raw float data temporarily in a ring/arena
    // to decouple the client from the quantizer.
    float data[512];
};

class IngestionActor
{
public:
    // HOT PATH: Wait-free or low-contention
    void submit(uint64_t label, const std::vector<float> &vec)
    {
        // 1. Acquire slot in current active buffer
        auto *slot = active_buffer_.reserve_slot();

        // 2. Memcpy (Fast, sequential RAM access)
        slot->label = label;
        std::memcpy(slot->data, vec.data(), dim_ * sizeof(float));

        // 3. If buffer full, signal background thread & swap
        if (active_buffer_.is_full())
        {
            wake_worker();
        }
    }

private:
    // BACKGROUND PATH: The "Merger"
    void worker_loop()
    {
        while (running_)
        {
            Batch *batch = wait_for_batch();

            // 1. Bulk Quantize (SIMD friendly now!)
            // We can quantize 1000 vectors at once.
            // Much faster than 1 at a time.
            quantizer_->encode_batch(batch->vectors, out_codes);

            // 2. Group by Centroid (Routing)
            // Instead of locking Shard A, then Shard B, then Shard A...
            // We group all inserts for Shard A and lock it ONCE.
            auto grouped = route_batch(batch, out_codes);

            // 3. Bulk Insert to Shards
            for (auto &[shard_id, items] : grouped)
            {
                // Lock held for minimum time, strictly for memory append
                shards_[shard_id]->bulk_append(items);
            }
        }
    }
};