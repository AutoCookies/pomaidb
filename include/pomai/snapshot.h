#pragma once

namespace pomai
{
    // Opaque handle for a point-in-time view of the database.
    // Thread-safe and immutable.
    class Snapshot
    {
    public:
        virtual ~Snapshot() = default;
    };

} // namespace pomai
