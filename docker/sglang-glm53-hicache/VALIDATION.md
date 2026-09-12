# Validation state

The advice-only test image allocated all TP8 host pools and passed short chat
and tool-call checks, but a 230k-token write-through request drove per-GPU HBM
from roughly 115 GiB to 141.3 GiB and stopped progressing. This image revision
adds an explicit synchronized CPU prefetch before exposing each host tensor.

The allocator, CPU-prefetch path, and failure cleanup apply cleanly to exact SGLang source
`0bcd822377da7b5718e674eaf9c870d349424dd1`. CPU unit tests are enforced in
the publish workflow. Runtime acceptance still requires managed allocation,
CPU residency on all eight ranks, a real eviction, and a host restore.
