# Vendored SAM code

Author: Lucien  
Date: 2026-09-23

`src/v2_sam`, `src/v3_sam`, and `src/v3p1_sam` come from [muggled_sam](https://github.com/heyoeyo/muggled_sam) commit `80d85ff` (heyoeyo, 2024). Those trees stay under the Apache License 2.0. See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) and [LICENSES/Apache-2.0.txt](../LICENSES/Apache-2.0.txt).

Local patches on that baseline:

- The position-encoding cache comparison no longer uses a chained `in` test (`src/v2_sam/components/posenc_sine.py`, `src/v3_sam/components/position_encoding.py`, `src/v3p1_sam/components/position_encoding.py`).
- SAM 2 memory RoPE caches the token grid shape, not only the token count (`src/v2_sam/components/memory_image_fusion_attention.py`).
- SAM 3 and SAM 3.1 scale each RoPE axis with the matching token dimension (the two `position_encoding.py` files above).

SAM 3.1 `step_video_masking` already calls `step_video_masking_multiplex` with one slot. This app still tracks each object through its own memory bank. Batching those banks into groups of 16 is not enabled: it would change results and needs a benchmark before it becomes a switch.
