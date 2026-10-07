import inspect
from sglang.srt.mem_cache import unified_radix_cache as urc
from sglang.srt.mem_cache.unified_cache import unified_tree_core as utc

tree_src = inspect.getsource(utc)
assert "self.hash_page_size = params.page_size" in tree_src
assert "compute_node_hash_values(node, self.hash_page_size)" in tree_src
assert "compute_node_hash_values(new_node, self.hash_page_size)" in tree_src
cache_src = inspect.getsource(urc)
assert "self.tree_core.hash_page_size = self._transfer_page_size" in cache_src
assert 'reason="tree_page_align"' in cache_src
# The startup guard is lifted (cache host-memory mode); attaching a store at runtime stays refused.
assert "storage hashes and transfers require matching page sizes" not in cache_src
assert "Compressed DSA storage supports --hicache-host-memory-mode cache only." in cache_src
print("4/5 storage hash chain at the transfer page")
