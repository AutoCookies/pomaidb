/* ----------------------------------------------------------------------------
Copyright (c) 2019-2023 Microsoft Research, Daan Leijen
This is free software; you can redistribute it and/or modify it under the
terms of the MIT license. A copy of the license can be found in the file
"LICENSE" at the root of this distribution.
-----------------------------------------------------------------------------*/

/* ----------------------------------------------------------------------------
Concurrent bitmap that can set/reset sequences of bits atomically,
represented as an array of fields where each field is a machine word (`size_t`)

There are two api's; the standard one cannot have sequences that cross
between the bitmap fields (and a sequence must be <= PALLOC_BITMAP_FIELD_BITS).
(this is used in region allocation)

The `_across` postfixed functions do allow sequences that can cross over
between the fields. (This is used in arena allocation)
---------------------------------------------------------------------------- */
#pragma once
#ifndef PALLOC_BITMAP_H
#define PALLOC_BITMAP_H

/* -----------------------------------------------------------
  Bitmap definition
----------------------------------------------------------- */

#define PALLOC_BITMAP_FIELD_BITS   (8*PALLOC_SIZE_SIZE)
#define PALLOC_BITMAP_FIELD_FULL   (~((size_t)0))   // all bits set

// An atomic bitmap of `size_t` fields
typedef _Atomic(size_t)  palloc_bitmap_field_t;
typedef palloc_bitmap_field_t*  palloc_bitmap_t;

// A bitmap index is the index of the bit in a bitmap.
typedef size_t palloc_bitmap_index_t;

// Create a bit index.
static inline palloc_bitmap_index_t palloc_bitmap_index_create_ex(size_t idx, size_t bitidx) {
  palloc_assert_internal(bitidx <= PALLOC_BITMAP_FIELD_BITS);
  return (idx*PALLOC_BITMAP_FIELD_BITS) + bitidx;
}
static inline palloc_bitmap_index_t palloc_bitmap_index_create(size_t idx, size_t bitidx) {
  palloc_assert_internal(bitidx < PALLOC_BITMAP_FIELD_BITS);
  return palloc_bitmap_index_create_ex(idx,bitidx);
}

// Create a bit index.
static inline palloc_bitmap_index_t palloc_bitmap_index_create_from_bit(size_t full_bitidx) {  
  return palloc_bitmap_index_create(full_bitidx / PALLOC_BITMAP_FIELD_BITS, full_bitidx % PALLOC_BITMAP_FIELD_BITS);
}

// Get the field index from a bit index.
static inline size_t palloc_bitmap_index_field(palloc_bitmap_index_t bitmap_idx) {
  return (bitmap_idx / PALLOC_BITMAP_FIELD_BITS);
}

// Get the bit index in a bitmap field
static inline size_t palloc_bitmap_index_bit_in_field(palloc_bitmap_index_t bitmap_idx) {
  return (bitmap_idx % PALLOC_BITMAP_FIELD_BITS);
}

// Get the full bit index
static inline size_t palloc_bitmap_index_bit(palloc_bitmap_index_t bitmap_idx) {
  return bitmap_idx;
}

/* -----------------------------------------------------------
  Claim a bit sequence atomically
----------------------------------------------------------- */

// Try to atomically claim a sequence of `count` bits in a single
// field at `idx` in `bitmap`. Returns `true` on success.
bool _palloc_bitmap_try_find_claim_field(palloc_bitmap_t bitmap, size_t idx, const size_t count, palloc_bitmap_index_t* bitmap_idx);

// Starts at idx, and wraps around to search in all `bitmap_fields` fields.
// For now, `count` can be at most PALLOC_BITMAP_FIELD_BITS and will never cross fields.
bool _palloc_bitmap_try_find_from_claim(palloc_bitmap_t bitmap, const size_t bitmap_fields, const size_t start_field_idx, const size_t count, palloc_bitmap_index_t* bitmap_idx);

// Like _palloc_bitmap_try_find_from_claim but with an extra predicate that must be fullfilled
typedef bool (palloc_cdecl *palloc_bitmap_pred_fun_t)(palloc_bitmap_index_t bitmap_idx, void* pred_arg);
bool _palloc_bitmap_try_find_from_claim_pred(palloc_bitmap_t bitmap, const size_t bitmap_fields, const size_t start_field_idx, const size_t count, palloc_bitmap_pred_fun_t pred_fun, void* pred_arg, palloc_bitmap_index_t* bitmap_idx);

// Set `count` bits at `bitmap_idx` to 0 atomically
// Returns `true` if all `count` bits were 1 previously.
bool _palloc_bitmap_unclaim(palloc_bitmap_t bitmap, size_t bitmap_fields, size_t count, palloc_bitmap_index_t bitmap_idx);

// Try to set `count` bits at `bitmap_idx` from 0 to 1 atomically. 
// Returns `true` if successful when all previous `count` bits were 0.
bool _palloc_bitmap_try_claim(palloc_bitmap_t bitmap, size_t bitmap_fields, size_t count, palloc_bitmap_index_t bitmap_idx);

// Set `count` bits at `bitmap_idx` to 1 atomically
// Returns `true` if all `count` bits were 0 previously. `any_zero` is `true` if there was at least one zero bit.
bool _palloc_bitmap_claim(palloc_bitmap_t bitmap, size_t bitmap_fields, size_t count, palloc_bitmap_index_t bitmap_idx, bool* any_zero);

bool _palloc_bitmap_is_claimed(palloc_bitmap_t bitmap, size_t bitmap_fields, size_t count, palloc_bitmap_index_t bitmap_idx);
bool _palloc_bitmap_is_any_claimed(palloc_bitmap_t bitmap, size_t bitmap_fields, size_t count, palloc_bitmap_index_t bitmap_idx);


//--------------------------------------------------------------------------
// the `_across` functions work on bitmaps where sequences can cross over
// between the fields. This is used in arena allocation
//--------------------------------------------------------------------------

// Find `count` bits of zeros and set them to 1 atomically; returns `true` on success.
// Starts at idx, and wraps around to search in all `bitmap_fields` fields.
bool _palloc_bitmap_try_find_from_claim_across(palloc_bitmap_t bitmap, const size_t bitmap_fields, const size_t start_field_idx, const size_t count, palloc_bitmap_index_t* bitmap_idx);

// Set `count` bits at `bitmap_idx` to 0 atomically
// Returns `true` if all `count` bits were 1 previously.
bool _palloc_bitmap_unclaim_across(palloc_bitmap_t bitmap, size_t bitmap_fields, size_t count, palloc_bitmap_index_t bitmap_idx);

// Set `count` bits at `bitmap_idx` to 1 atomically
// Returns `true` if all `count` bits were 0 previously. `any_zero` is `true` if there was at least one zero bit.
bool _palloc_bitmap_claim_across(palloc_bitmap_t bitmap, size_t bitmap_fields, size_t count, palloc_bitmap_index_t bitmap_idx, bool* pany_zero, size_t* already_set);

bool _palloc_bitmap_is_claimed_across(palloc_bitmap_t bitmap, size_t bitmap_fields, size_t count, palloc_bitmap_index_t bitmap_idx, size_t* already_set);
bool _palloc_bitmap_is_any_claimed_across(palloc_bitmap_t bitmap, size_t bitmap_fields, size_t count, palloc_bitmap_index_t bitmap_idx);

#endif
