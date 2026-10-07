import importlib
for mod in (
    "sglang.srt.mem_cache.unified_radix_cache",
    "sglang.srt.mem_cache.unified_cache.unified_tree_core",
    "sglang.srt.mem_cache.startup_ram_budget",
    "sglang.srt.mem_cache.hicache_storage",
    "sglang.srt.mem_cache.storage.file.lru_file_evictor",
):
    importlib.import_module(mod)
print("1/5 patched modules import")
