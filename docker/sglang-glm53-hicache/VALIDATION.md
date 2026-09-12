# Validation state

Pending test-only image publication and gpu23 PPCIe qualification.

The allocator and its failure cleanup apply cleanly to exact SGLang source
`0bcd822377da7b5718e674eaf9c870d349424dd1`. CPU unit tests are enforced in
the publish workflow. Runtime acceptance requires managed allocation/advice on
all eight ranks followed by a real eviction and host restore.
