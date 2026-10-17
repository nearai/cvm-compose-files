# Start-up patches for stock SGLang v0.5.21 (lmsysorg/sglang:v0.5.21-cu130) serving GLM-5.3-Flash W4AFP8.
# 1) W4AFP8 loader: from_config drops modules_to_not_convert, so narrow layers get FP8 block-quant and
#    fail ("output_partition_size = 16 is not divisible by block_n = 128"). Same fix as
#    docker/sglang-glm53-hicache-w4afp8/modules-to-not-convert.diff.
# 2) Thinking budget: accept the public thinking_token_budget and default GLM5 (glm5_next) to 8192,
#    the production policy (fork 1fea50d9f + e4d313ef0). Without it reasoning runs uncapped.
# 3) P/D under CC: MetadataBuffers are pageable CPU tensors that NIXL registers as DRAM; UCX's
#    cuda_copy MD then calls cuMemHostRegister, which HCC/PPCIe rejects (NIXL_ERR_BACKEND). Pinning
#    them via cudaHostAlloc (torch pin_memory) makes UCX skip the register (gpu03, 2026-10-06).
# 4) Decode TP-rank divergence guard: under overload with client aborts the decode transfer queue
#    can differ by a request between TP ranks; the MIN all-reduce of poll states then mismatches
#    in size and gloo aborts the engine ("op.preamble.length <= op.nbytes. 16 vs 15", gpu03
#    2026-10-06, also seen in the lab). Poll only requests queued on every rank, in rid order;
#    the rest stay Transferring until their peers catch up.
# Each edit is an exact-anchor replacement; any missing anchor aborts the start (fail closed).
import os, pathlib, sys
ROOT = pathlib.Path(os.environ.get("PD_PATCH_ROOT", "/sgl-workspace/sglang/python/sglang/srt"))
def sub(rel, old, new, label):
    p = ROOT / rel; s = p.read_text()
    if new in s:
        print(f"[pd-patch] {label}: already applied", flush=True); return
    n = s.count(old)
    if n != 1:
        sys.exit(f"[pd-patch] {label}: expected 1 anchor, found {n}")
    p.write_text(s.replace(old, new)); print(f"[pd-patch] {label}: applied", flush=True)
sub("layers/quantization/w4afp8.py",
"""        weight_block_size = [128, 128]
        return cls(
            is_checkpoint_fp8_serialized=is_checkpoint_fp8_serialized,
            is_checkpoint_w4afp8_serialized=is_checkpoint_w4afp8_serialized,
            linear_activation_scheme=linear_activation_scheme,
            moe_activation_scheme=moe_activation_scheme,
            weight_block_size=weight_block_size,
        )""",
"""        weight_block_size = [128, 128]
        ignored_layers = (
            config.get("modules_to_not_convert")
            or config.get("ignored_layers")
            or config.get("ignore")
            or []
        )
        return cls(
            is_checkpoint_fp8_serialized=is_checkpoint_fp8_serialized,
            is_checkpoint_w4afp8_serialized=is_checkpoint_w4afp8_serialized,
            linear_activation_scheme=linear_activation_scheme,
            moe_activation_scheme=moe_activation_scheme,
            ignored_layers=ignored_layers,
            weight_block_size=weight_block_size,
        )""", "w4afp8 modules_to_not_convert")
sub("entrypoints/openai/protocol.py", "    StrictBool,\n    field_serializer,", "    StrictBool,\n    StrictInt,\n    field_serializer,", "protocol StrictInt import")
sub("entrypoints/openai/protocol.py",
"""    custom_params: Optional[Dict] = None

    # Pre-computed prompt token IDs: when provided""",
"""    custom_params: Optional[Dict] = None
    thinking_token_budget: Optional[StrictInt] = Field(default=None, ge=0)

    # Pre-computed prompt token IDs: when provided""", "protocol thinking_token_budget field")
sub("entrypoints/openai/protocol.py",
"""    @model_validator(mode="before")
    @classmethod
    def set_tool_choice_default(cls, values):""",
"""    @model_validator(mode="before")
    @classmethod
    def normalize_thinking_token_budget(cls, values):
        if not isinstance(values, dict):
            return values
        custom_params = values.get("custom_params")
        if custom_params is None:
            custom_params = {}
        if not isinstance(custom_params, dict) or "thinking_budget" in custom_params:
            return values
        if "thinking_token_budget" in values:
            budget = values.get("thinking_token_budget")
        else:
            return values
        if isinstance(budget, int) and not isinstance(budget, bool) and budget >= 0:
            values = dict(values)
            custom_params = dict(custom_params)
            custom_params["thinking_budget"] = budget
            values["custom_params"] = custom_params
        return values

    @model_validator(mode="before")
    @classmethod
    def set_tool_choice_default(cls, values):""", "protocol thinking_token_budget normalizer")
sub("entrypoints/openai/serving_chat.py",
"""        set_request_reasoning_end_token_ids(
            sampling_params, processed_messages.reasoning_end_token_ids
        )""",
"""        if (
            getattr(self.tokenizer_manager.model_config.hf_config, "model_type", None)
            in ("glm5_next", "glm5_next_text")
            and "thinking_token_budget" not in request.model_fields_set
        ):
            custom_params = dict(sampling_params.get("custom_params") or {})
            custom_params.setdefault("thinking_budget", 8192)
            sampling_params["custom_params"] = custom_params
        set_request_reasoning_end_token_ids(
            sampling_params, processed_messages.reasoning_end_token_ids
        )""", "serving_chat glm5 default thinking budget 8192")
sub("disaggregation/utils.py",
"""                    device=self.bootstrap_room.device,
                )

    def set_kv_checksum(""",
"""                    device=self.bootstrap_room.device,
                )
        if torch.cuda.is_available():
            for _name, _t in list(vars(self).items()):
                if isinstance(_t, torch.Tensor) and _t.device.type == "cpu":
                    setattr(self, _name, _t.pin_memory())

    def set_kv_checksum(""", "disagg MetadataBuffers pinned host memory")
sub("disaggregation/decode.py",
"""    def _poll_with_metadata_gate(self) -> List[int]:
        pollers = (""",
"""    def _poll_with_metadata_gate(self) -> List[int]:
        group = self.gloo_group
        if (
            not self.scheduler.enable_decode_hicache
            and torch.distributed.get_world_size(group) > 1
        ):
            rids = [dr.req.rid for dr in self.queue]
            gathered = [None] * torch.distributed.get_world_size(group)
            torch.distributed.all_gather_object(gathered, rids, group=group)
            common = set(rids).intersection(*(set(g) for g in gathered))
            if any(len(g) != len(common) for g in gathered):
                logger.warning(
                    f"[pd-patch] decode transfer queue diverged across TP ranks: "
                    f"{[len(g) for g in gathered]} queued, {len(common)} common"
                )
            subset = sorted(
                (dr for dr in self.queue if dr.req.rid in common),
                key=lambda dr: dr.req.rid,
            )
            sub_polls = poll_and_all_reduce(
                [dr.kv_receiver for dr in subset],
                group,
                decode_reqs=subset,
                metadata_buffers=self.metadata_buffers,
            ) if subset else []
            by_rid = {dr.req.rid: poll for dr, poll in zip(subset, sub_polls)}
            return [by_rid.get(dr.req.rid, KVPoll.Transferring) for dr in self.queue]
        pollers = (""", "decode TP-rank divergence guard")
