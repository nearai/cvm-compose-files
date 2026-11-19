"""pytest plugin: stub sgl_kernel's compiled ops so CPU-only containers can collect the upstream unit tests."""
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()
