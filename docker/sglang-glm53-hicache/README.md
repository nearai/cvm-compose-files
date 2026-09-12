# GLM-5.3 non-Flash HiCache HCC canary image

This experiment layers one opt-in host allocator onto the exact SGLang image
used by the gpu23 control. The base image contains SGLang
`0bcd822377da7b5718e674eaf9c870d349424dd1`; build-time checks reject any
different source bytes.

Set `SGLANG_HICACHE_CUDA_MANAGED_MEMORY=1` only in a CUDA confidential guest
where `cudaHostRegister` is unsupported. The allocator uses
`cudaMallocManaged`, sets CPU-preferred residency, grants access to the
current rank GPU, explicitly prefetches the range to CPU with the CUDA 13
location API, synchronizes that migration, wraps the range as the CPU tensor
expected by HiCache, and records that it must not be host-unregistered. Advice,
prefetch, or synchronization failure frees the range and aborts startup. The
default allocator is unchanged without the opt-in.

The managed path accepts only SGLang's default in-process host allocator and
rejects SHM or external storage allocators. The first gpu23 candidate uses
`write_through`, `kernel`, and `page_first`; direct/pooled transfers remain
out of scope until kernel restore is proven.

This allocator does not by itself establish correctness. Qualification
requires a forced HBM eviction, nonzero host load-back counters, identical
deterministic output after restore, bounded HBM/RAM use, and no XID/ECC/OOM.

The explicit CPU prefetch is required on gpu23: advice alone left untouched
managed pages resident in HBM, and the first 230k-token write-through request
filled HBM before completing. Prefetch is therefore part of the allocator's
startup contract rather than an optional performance hint.
