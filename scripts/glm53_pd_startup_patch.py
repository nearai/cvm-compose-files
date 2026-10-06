# Start-up patches for stock SGLang v0.5.21 (lmsysorg/sglang:v0.5.21-cu130) serving GLM-5.3-Flash W4AFP8.
# 1) W4AFP8 loader: from_config drops modules_to_not_convert, so narrow layers get FP8 block-quant and
#    fail ("output_partition_size = 16 is not divisible by block_n = 128"). Same fix as
#    docker/sglang-glm53-hicache-w4afp8/modules-to-not-convert.diff.
# 2) Thinking budget: accept the public thinking_token_budget and default GLM5 (glm5_next) to 8192,
#    the production policy (fork 1fea50d9f + e4d313ef0). Without it reasoning runs uncapped.
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
