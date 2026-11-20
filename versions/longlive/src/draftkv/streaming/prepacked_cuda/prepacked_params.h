#pragma once
#include "src/flash.h"
// No per-thread copy of Flash_fwd_params: the entire immutable parameter block is
// __grid_constant__. Group corrections are applied only at the K/V base address.
struct PreparedParams : flash::Flash_fwd_params {
    const int64_t *token_offsets;
    const int *time_slots;
    const double *time_phase;
    bool chrono;
    int kv_head_group;
    int64_t k_group_stride;
    int64_t v_group_stride;
};
