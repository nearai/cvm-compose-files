#!/usr/bin/env python3
"""Lab copy of the gpu02 DS4.1 pack's shape for a bare-metal 4-GPU host (one replica).

Four per-rank vLLM containers with the exact engine argv of scripts/prepare_dsv41_lora_dep4x2.py,
plus the two strict-model proxies in front (base on 127.0.0.1:31150, abliterated on 31151); rank
metrics on 127.0.0.1:31160-31163. No CVM pieces (dstack, certs, registrar, TLS).

  lab_dsv41_compose_shape.py ENGRAM(host|hbm) ENGINE_IMAGE PROXY_IMAGE HUB_DIR ADAPTER_DIR > compose.yaml
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_dsv41_lora_dep4x2 as pack  # noqa: E402

engram, engine_image, proxy_image, hub_dir, adapter_dir = sys.argv[1:6]
pack.ENGRAM_LAYOUT = engram
replica = "r1"
adapter_mount = pack.ADAPTER_PATH
base_repo = f"{pack.HF_HUB}/models--deepseek-ai--DeepSeek-V4.1-Flash-1017"
ranks = [pack.rank_service(replica, i) for i in range(4)]

out = ["services:\n"]
for rank, svc in enumerate(ranks):
    command = pack.engine_command(replica, rank).replace("\n      ", " ")
    depends = f"    depends_on: [{ranks[0]}]\n" if rank else ""
    out.append(f"""  {svc}:
    image: {engine_image}
    container_name: {svc}
    init: true
    ipc: host
    shm_size: 64g
    ulimits: {{memlock: -1, nofile: {{soft: 65535, hard: 65535}}}}
{depends}    command: >
      {command}
    environment:
      - HF_HUB_OFFLINE=1
      - PYTHONUNBUFFERED=1
      - NCCL_DEBUG=WARN
      - 'VLLM_HUMMING_INPUT_QUANT_CONFIG={{"dtype":"float8e4m3","quant_mode":"dynamic_token"}}'
    volumes:
      - {hub_dir}:{base_repo}:ro
      - {adapter_dir}:{adapter_mount}:ro
      - /data/dsv41-lora-lab-20261026/cache/vllm:/root/.cache/vllm
    ports: ["127.0.0.1:{31160 + rank}:8000"]
    cpuset: "24-43,120-139"
    mem_limit: 1000g
    deploy:
      resources:
        reservations:
          devices: [{{driver: nvidia, device_ids: ["{rank}"], capabilities: [gpu]}}]
    labels: {{nearai.experiment: dsv41lora, nearai.arm: l12}}
""")
backends = ",".join(f"http://{svc}:8000" for svc in ranks)
for (name, model, *_), port in zip(pack.PROXIES, (31150, 31151)):
    out.append(f"""  {name}:
    image: {proxy_image}
    container_name: lab-{name}
    environment:
      - MODEL_NAME={model}
      - VLLM_PROXY_STRICT_MODEL=1
      - TOKEN=${{LAB_TOKEN}}
      - VLLM_BACKEND_URLS={backends}
      - VLLM_BACKEND_CONVERSATION_AFFINITY=1
      - VLLM_PROXY_MAX_REQUEST_SIZE=104857600
      - VLLM_PROXY_POOL_IDLE_TIMEOUT_SECS=3
      - NON_TEE_DEPLOYMENT=1
      - DEV=1
      - GPU_NO_HW_MODE=1
      - LISTEN_ADDR=0.0.0.0
      - LISTEN_PORT=8000
      - LOG_FORMAT=json
    ports: ["127.0.0.1:{port}:8000"]
    cpuset: "44-45"
    labels: {{nearai.experiment: dsv41lora, nearai.arm: l12}}
""")
sys.stdout.write("".join(out))
