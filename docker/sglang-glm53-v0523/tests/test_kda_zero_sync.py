"""v0.5.23 replacement for the kda-zero-sync.diff CPU check. The patch is DROPPED: upstream #39688
(c8eb54c41d, "Fuse GLM-5.3-Flash KDA projections and prefill metadata") computes the logical token
count once per batch on the host in HybridLinearAttnBackend metadata (sum(extend_seq_lens_cpu)), and
KDAAttnBackend.forward_extend reads forward_metadata.logical_num_tokens, falling back to the device
read only when it is None. This checks both halves by ast so a regression back to a per-layer
int(query_start_loc[-1]) sync is caught.

Usage: python3 -I test_kda_zero_sync.py .../layers/attention/linear/kda_backend.py .../layers/attention/hybrid_linear_attn_backend.py
"""

import ast
import sys

kda, hybrid = sys.argv[1], sys.argv[2]
tree = ast.parse(open(kda).read())
fe = next(n for c in tree.body if isinstance(c, ast.ClassDef) and c.name == "KDAAttnBackend"
          for n in c.body if isinstance(n, ast.FunctionDef) and n.name == "forward_extend")
src = ast.unparse(fe)
assert "logical_num_tokens = self.forward_metadata.logical_num_tokens" in src, "host count not read from metadata"
reads = [n for n in ast.walk(fe) if isinstance(n, ast.Call) and ast.unparse(n) == "int(query_start_loc[-1])"]
assert len(reads) == 1, len(reads)
guard = next(n for n in ast.walk(fe) if isinstance(n, ast.If) and ast.unparse(n.test) == "logical_num_tokens is None")
assert any(ast.unparse(r) in ast.unparse(guard) for r in reads), "device read must be only the None fallback"

hsrc = open(hybrid).read()
assert "logical_num_tokens = sum(forward_batch.extend_seq_lens_cpu)" in hsrc, "metadata host count missing"
print("kda host token count is upstream (#39688): checks passed")
