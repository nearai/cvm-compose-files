"""stack: engine isolation, proxy wiring, registrar auth and OTel labels, from the un-interpolated render.

The checks read the render with `interpolate=False`: anchors are resolved, while the deploy-time
markers and `$$` escapes stay as written.
"""

import copy
import json
import re

from common import SERVICES, Finding, YamlError, parse_yaml

RULES = [
    "stack.engine-network", "stack.engine-member", "stack.engine-backend", "stack.engine-pending",
    "stack.proxy-depends", "stack.proxy-env", "stack.registrar-auth",
    "stack.otel-collector", "stack.otel-logs", "stack.otel-scrape",
]
DIFF_RULES = []

ENGINE_IMAGE = re.compile(r"^(docker\.io/)?(lmsysorg/sglang|vllm/vllm-openai|nearaidev/sglang|nearaidev/vllm)[@:]")
PROXY_IMAGE = re.compile(r"^(docker\.io/)?nearaidev/vllm-proxy-rs[@:]")
PROXY_IMAGE_PREFIX = "nearaidev/vllm-proxy-rs"
COLLECTOR_IMAGE = re.compile(r"^(docker\.io/)?otel/opentelemetry-collector-contrib[@:]")
PROBE_IMAGE = re.compile(r"^(docker\.io/)?curlimages/curl[@:]")
BACKEND_ENV = ("VLLM_BACKEND_URLS", "VLLM_BASE_URL")
BACKEND_URL = re.compile(r"(?:\w+://)?(\[[^\]]+\]|[^:/\[\]$]+)(?::\d+)?(?:/.*)?")
LOOPBACK = re.compile(r"(127\.0\.0\.1|localhost):\d+")
FORBIDDEN_ENV = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN")

REQUIRED_PROXY_ENV = ("WEB_CONTEXT_SEARCH_URL", "WEB_CONTEXT_SEARCH_API_KEY")
REQUIRED_PROD_PROXY_ENV = {
    "VLLM_PROXY_INLINE_MEDIA": "${VLLM_PROXY_INLINE_MEDIA:-1}",
    "VLLM_PROXY_REJECTED_CONTENT_PART_TYPES": "${VLLM_PROXY_REJECTED_CONTENT_PART_TYPES:-video_url}",
}
PROXY_TOKEN_ENV = "${PROXY_TOKEN}"
AUTH_HEADER = "Authorization: Bearer $$INFERENCE_TOKEN"
INFERENCE_TOKEN = 'INFERENCE_TOKEN="$${PROXY_TOKEN}"'

REQUIRED_COLLECTOR_ENV = ("MONITORING_INGEST_TOKEN", "CVM_NAME", "CVM_HOST", "DD_HOSTNAME", "ENV", "HOST_IP")
DATADOG_AGENT_CHECK_LABELS = ("com.datadoghq.ad.check_names", "com.datadoghq.ad.init_configs",
                              "com.datadoghq.ad.instances")
REQUIRED_LOG_TAGS = ("deployment", "env", "host", "ip")
REQUIRED_OTEL_SCRAPE_LABELS = (
    "nearai.otel.service", "nearai.otel.source", "nearai.otel.container_name", "nearai.otel.model",
    "nearai.otel.deployment", "nearai.otel.env", "nearai.otel.host", "nearai.otel.host_machine",
    "nearai.otel.cvm_name", "nearai.otel.ip", "nearai.otel.port", "nearai.otel.path",
)
OPTIONAL_SCRAPE_TAGS = ("model_path", "precision", "instance", "replica")


def _s(v):
    return "" if v is None else str(v)


def as_list(v):
    if v is None:
        return []
    return v if isinstance(v, list) else [v]


def env_of(svc):
    """Environment as {name: value}; a value is None when the entry has no `=`."""
    env = svc.get("environment") or []
    if isinstance(env, dict):
        return {k: None if v is None else str(v) for k, v in env.items()}
    out = {}
    for entry in env:
        key, sep, value = str(entry).partition("=")
        out[key] = value if sep else None
    return out


def nets_of(svc):
    nets = svc.get("networks")
    if isinstance(nets, dict):
        return [str(k) for k in nets]
    return [str(n) for n in as_list(nets)]


def image_of(svc):
    return _s(svc.get("image"))


def config_text(model, name):
    return ((model.get("configs") or {}).get(name) or {}).get("content")


def config_sources(svc):
    out = []
    for entry in as_list(svc.get("configs")):
        source = entry.get("source") if isinstance(entry, dict) else entry
        if source is not None:
            out.append(str(source))
    return out


def load_config(text):
    """Parse an embedded collector or scrape config; `$$` is compose's escape for `$`."""
    return parse_yaml(text.replace("$$", "$"))


# ---------------------------------------------------------------- engine isolation

def proxy_backends(path, services, out):
    """Service names that proxies send requests to, through VLLM_BACKEND_URLS or VLLM_BASE_URL."""
    by_container = {_s(s.get("container_name")): n for n, s in services.items()}
    found = []
    for name, svc in services.items():
        if not PROXY_IMAGE.match(image_of(svc)):
            continue
        env = env_of(svc)
        for key in BACKEND_ENV:
            if not env.get(key):
                continue
            for raw in env[key].split(","):
                url = raw.strip()
                m = BACKEND_URL.fullmatch(url)
                host = m.group(1) if m else None
                backend = host if host in services else by_container.get(host)
                if backend is None:
                    out.append(Finding(path, 0, "stack.engine-backend",
                                       f"services.{name}: backend URL {url!r} does not name a service in this file",
                                       key=f"{name}:{url}"))
                else:
                    found.append(backend)
    return found


def engine_errors(svc, networks):
    out = []
    nets = nets_of(svc)
    if not nets:
        out.append("must list its networks explicitly")
    if "network_mode" in svc:
        out.append("network_mode is not allowed")
    if "ports" in svc:
        out.append("must not publish ports")
    for net in nets:
        if (networks.get(net) or {}).get("internal") is not True:
            out.append(f"network {net} is not internal")
    env = env_of(svc)
    for key in FORBIDDEN_ENV:
        if key in env:
            out.append(f"{key} must not be set")
    if env.get("HF_HUB_OFFLINE") != "1":
        out.append("HF_HUB_OFFLINE must be the literal 1")
    return out


def otlp_endpoints(config):
    receivers = config.get("receivers") if isinstance(config, dict) else None
    out = []
    for name, receiver in (receivers or {}).items():
        if _s(name).split("/")[0] != "otlp":
            continue
        for protocol, settings in (((receiver or {}).get("protocols")) or {}).items():
            out.append((protocol, _s((settings or {}).get("endpoint"))))
    return out


def collector_configs(model, svc, out, path, name):
    configs = []
    for source in config_sources(svc):
        text = config_text(model, source)
        if not text:
            continue
        try:
            configs.append(load_config(text))
        except YamlError:
            out.append(Finding(path, 0, "stack.engine-member",
                               f"services.{name}: collector config {source} is not readable",
                               key=f"{name}:{source}"))
    return configs


def check_members(path, model, services, engines, engine_nets, out):
    for name, svc in services.items():
        if name in engines:
            continue
        shared = [n for n in nets_of(svc) if n in engine_nets]
        if not shared:
            continue
        where = f"{path}: services.{name}"
        image = image_of(svc)
        if PROXY_IMAGE.match(image):
            continue
        if COLLECTOR_IMAGE.match(image):
            for config in collector_configs(model, svc, out, path, name):
                for protocol, endpoint in otlp_endpoints(config):
                    if not LOOPBACK.fullmatch(endpoint):
                        out.append(Finding(path, 0, "stack.engine-member",
                                           f"{where}: OTLP {protocol} receiver must listen on loopback",
                                           key=f"{where}:{protocol}"))
        elif PROBE_IMAGE.match(image):
            if any(n not in engine_nets for n in nets_of(svc)):
                out.append(Finding(path, 0, "stack.engine-member",
                                   f"{where}: the probe must join engine networks only", key=f"{where}:probe"))
        else:
            out.append(Finding(path, 0, "stack.engine-member",
                               f"{where}: not allowed on engine network {', '.join(shared)}", key=f"{where}:net"))


def check_isolation(path, model, pending, out):
    services = model.get("services") or {}
    networks = model.get("networks") or {}
    backends = set(proxy_backends(path, services, out))
    detected = {n: s for n, s in services.items() if ENGINE_IMAGE.match(image_of(s)) or n in backends}
    for name in sorted(pending):
        if name not in detected:
            out.append(Finding(path, 0, "stack.engine-pending",
                               f"{path}: pending service {name} is not an engine in this file", key=name))
        elif not engine_errors(detected[name], networks):
            out.append(Finding(path, 0, "stack.engine-pending",
                               f"{path}: pending service {name} follows the contract now; remove its pending mark",
                               key=name))
    if not detected:
        out.append(Finding(path, 0, "stack.engine-network", f"{path}: no engine service found", key="none"))
    engines = {n: s for n, s in detected.items() if n not in pending}
    engine_nets = {net for s in engines.values() for net in nets_of(s)}
    check_members(path, model, services, engines, engine_nets, out)
    for name, svc in sorted(engines.items()):
        for msg in engine_errors(svc, networks):
            out.append(Finding(path, 0, "stack.engine-network", f"{path}: services.{name}: {msg}",
                               key=f"{name}:{msg}"))


# ---------------------------------------------------------------- proxies

def is_proxy_name(name):
    return name == "proxy-nginx" or name.startswith(("proxy-", "vllm-proxy-"))


def is_model_name(name):
    return name == "model-downloader" or name.startswith("model-")


def proxy_upstreams(model, svc):
    out = []
    for source in config_sources(svc):
        out += re.findall(r"proxy_pass\s+http://([A-Za-z0-9_.-]+)(?::\d+)?", config_text(model, source) or "")
    return list(dict.fromkeys(out))


def check_proxy_depends(path, model, services, out):
    for name, svc in services.items():
        deps_raw = svc.get("depends_on")
        deps = [str(d) for d in deps_raw] if isinstance(deps_raw, dict) else [str(d) for d in as_list(deps_raw)]
        if is_proxy_name(name):
            bad = [d for d in deps if is_model_name(d)]
            if bad:
                out.append(Finding(path, 0, "stack.proxy-depends",
                                   f"{path}: services.{name}.depends_on must not include model services: "
                                   f"{', '.join(bad)}", key=name))
        if name in ("nginx", "proxy-nginx"):
            missing = [u for u in proxy_upstreams(model, svc) if u in services and is_proxy_name(u) and u not in deps]
            if missing:
                out.append(Finding(path, 0, "stack.proxy-depends",
                                   f"{path}: services.{name}.depends_on missing proxy_pass upstreams: "
                                   f"{', '.join(missing)}", key=name))


def check_proxy_env(path, services, out):
    for name, svc in services.items():
        env = env_of(svc)
        is_proxy = ("MODEL_NAME" in env and "CLOUD_API_URL" in env) or \
            (path.startswith("prod/") and image_of(svc).startswith(PROXY_IMAGE_PREFIX))
        if not is_proxy:
            continue
        if path.startswith("prod/"):
            for var, expected in REQUIRED_PROD_PROXY_ENV.items():
                actual = env.get(var)
                if actual == expected:
                    continue
                problem = "is missing" if actual is None else f"is {actual!r}"
                out.append(Finding(path, 0, "stack.proxy-env",
                                   f'{path}: services.{name}.environment {var} {problem}, expected "{expected}"',
                                   key=f"{name}:{var}"))
        missing = [v for v in REQUIRED_PROXY_ENV if env.get(v) != "${" + v + "}"]
        if missing:
            out.append(Finding(path, 0, "stack.proxy-env",
                               f"{path}: services.{name}.environment missing {', '.join(missing)}", key=name))


# ---------------------------------------------------------------- registrar

def curl_blocks(script):
    lines = script.splitlines(keepends=True)
    blocks, i = [], 0
    while i < len(lines):
        if "curl " not in lines[i]:
            i += 1
            continue
        block = lines[i]
        while block.splitlines(keepends=True)[-1].rstrip().endswith("\\") and i + 1 < len(lines):
            i += 1
            block += lines[i]
        blocks.append(block)
        i += 1
    return blocks


def registrar_probe_blocks(script):
    ready_requires_auth = 'READY_URL="http://' in script
    return [b for b in curl_blocks(script)
            if "/v1/chat/completions" in b or "/v1/models" in b or (ready_requires_auth and "$$READY_URL" in b)]


def check_registrar(path, model, services, out):
    registrar = services.get("model-proxy-registrar")
    script = config_text(model, "registrar_script") or ""
    if registrar is None or not script:
        return
    blocks = registrar_probe_blocks(script)
    if not blocks:
        return
    if env_of(registrar).get("PROXY_TOKEN") != PROXY_TOKEN_ENV:
        out.append(Finding(path, 0, "stack.registrar-auth",
                           f"{path}: services.model-proxy-registrar.environment missing PROXY_TOKEN={PROXY_TOKEN_ENV}",
                           key="registrar-env"))
    if INFERENCE_TOKEN not in script:
        out.append(Finding(path, 0, "stack.registrar-auth",
                           f"{path}: configs.registrar_script.content missing {INFERENCE_TOKEN}",
                           key="registrar-token"))
    for i, block in enumerate(blocks):
        if AUTH_HEADER not in block:
            first = block.splitlines()[0].strip()
            out.append(Finding(path, 0, "stack.registrar-auth",
                               f"{path}: registrar inference probe {i + 1} missing {AUTH_HEADER}: {first}",
                               key=f"probe-{i}"))


# ---------------------------------------------------------------- OTel

def tag_map(tags):
    out = {}
    for tag in as_list(tags):
        key, sep, value = _s(tag).partition(":")
        if key and sep:
            out[key] = value
    return out


def public_ports_by_proxy(model):
    content = config_text(model, "nginx_conf") or ""
    out = {}
    for port, svc in re.findall(r"listen\s+(\d+);(?:(?!listen\s+\d+;).)*?proxy_pass\s+http:\/\/([^:;]+):8000;",
                                content, re.S):
        out.setdefault(svc, port)
    return out


def expected_public_ports(model, services):
    proxy_ports = public_ports_by_proxy(model)
    expected = dict(proxy_ports)
    for name, svc in services.items():
        if name not in proxy_ports:
            continue
        env = env_of(svc)
        urls = ",".join(env[k] for k in ("VLLM_BASE_URL", "VLLM_BACKEND_URLS") if env.get(k) is not None)
        for backend in re.findall(r"http:\/\/([^:\/,]+):\d+", urls):
            expected.setdefault(backend, proxy_ports[name])
    return expected


def check_otel_collector(path, model, services, out):
    collector = services.get("otelcol-contrib")
    volumes = model.get("volumes") or {}
    if collector is None:
        out.append(Finding(path, 0, "stack.otel-collector",
                           f"{path}: services.otelcol-contrib: missing app collector service", key="service"))
    text = config_text(model, "otelcol_app_config")
    if not text:
        out.append(Finding(path, 0, "stack.otel-collector",
                           f"{path}: configs.otelcol_app_config: missing app collector config", key="config"))
    if "otelcol_app_storage" not in volumes:
        out.append(Finding(path, 0, "stack.otel-collector",
                           f"{path}: volumes.otelcol_app_storage: missing app collector storage volume",
                           key="volume"))
    if collector is not None:
        env = env_of(collector)
        for var in REQUIRED_COLLECTOR_ENV:
            value = env.get(var)
            if value is None or "${" + var + ":-" not in value:
                out.append(Finding(path, 0, "stack.otel-collector",
                                   f"{path}: services.otelcol-contrib.environment: missing soft-default marker "
                                   f"for {var}", key=var))
    if not text:
        return
    try:
        cfg = load_config(text)
    except YamlError:
        out.append(Finding(path, 0, "stack.otel-collector",
                           f"{path}: configs.otelcol_app_config: invalid collector config", key="parse"))
        return
    where = f"{path}: configs.otelcol_app_config.exporters.otlphttp/gateway"
    exporter = ((cfg or {}).get("exporters") or {}).get("otlphttp/gateway")
    if exporter is None:
        out.append(Finding(path, 0, "stack.otel-collector", f"{where}: missing otlphttp/gateway exporter",
                           key="exporter"))
        return
    queue = exporter.get("sending_queue")
    if queue is None:
        out.append(Finding(path, 0, "stack.otel-collector", f"{where}.sending_queue: missing sending_queue configuration",
                           key="queue"))
        return
    consumers = queue.get("num_consumers")
    if isinstance(consumers, int) and not isinstance(consumers, bool) and consumers == 1:
        return
    if isinstance(consumers, str) and consumers.strip() == "1":
        return
    out.append(Finding(path, 0, "stack.otel-collector",
                       f"{where}.sending_queue.num_consumers: must be 1 to preserve cumulative metric request order",
                       key="consumers"))


def check_otel_logs(path, services, out):
    """Log labels per service; returns the parsed tags of every service that has one."""
    tags_by_service = {}
    for name, svc in services.items():
        labels = svc.get("labels") or {}
        for key in DATADOG_AGENT_CHECK_LABELS:
            if key in labels:
                out.append(Finding(path, 0, "stack.otel-logs",
                                   f"{path}: services.{name}.labels.{key}: Datadog Agent check labels are removed; "
                                   "use nearai.otel.scrape and an otelcol_app_config scrape target",
                                   key=f"{name}:{key}"))
        raw = labels.get("com.datadoghq.ad.logs")
        if raw is None:
            continue
        where = f"{path}: services.{name}.labels.com.datadoghq.ad.logs"
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            parsed = None
        if not (isinstance(parsed, list) and parsed and isinstance(parsed[0], dict)):
            out.append(Finding(path, 0, "stack.otel-logs", f"{where}: invalid JSON", key=f"{name}:json"))
            continue
        cfg = parsed[0]
        source, service_label = cfg.get("source"), cfg.get("service")
        tags = tag_map(cfg.get("tags"))
        if _s(source) == "":
            out.append(Finding(path, 0, "stack.otel-logs", f"{where}: missing source", key=f"{name}:source"))
        if _s(service_label) == "":
            out.append(Finding(path, 0, "stack.otel-logs", f"{where}: missing service", key=f"{name}:service"))
        for key in REQUIRED_LOG_TAGS:
            if key not in tags:
                out.append(Finding(path, 0, "stack.otel-logs", f"{where}: missing {key}: tag", key=f"{name}:{key}"))
        if name != "otelcol-contrib" and "model" not in tags:
            out.append(Finding(path, 0, "stack.otel-logs", f"{where}: missing model: tag", key=f"{name}:model"))
        otel_source = labels.get("nearai.otel.source")
        if otel_source is not None and source != otel_source:
            out.append(Finding(path, 0, "stack.otel-logs",
                               f"{where}: source {source!r} does not match nearai.otel.source {otel_source!r}",
                               key=f"{name}:otel-source"))
        otel_service = labels.get("nearai.otel.service")
        if otel_service is not None and service_label != otel_service:
            out.append(Finding(path, 0, "stack.otel-logs",
                               f"{where}: service {service_label!r} does not match nearai.otel.service "
                               f"{otel_service!r}", key=f"{name}:otel-service"))
        tags_by_service[name] = tags
    return tags_by_service


def scrape_targets(config):
    parsed = parse_yaml(config) or {}
    prom = ((parsed.get("receivers") or {}).get("prometheus/apps")) or {}
    targets = {}
    for scrape in as_list((prom.get("config") or {}).get("scrape_configs")):
        for static in as_list(scrape.get("static_configs")):
            for target in as_list(static.get("targets")):
                name = _s(target).split(":", 1)[0]
                targets[name] = {"target": target, "metrics_path": scrape.get("metrics_path"),
                                 "labels": static.get("labels") or {}}
    return targets


def check_otel_scrape(path, model, services, tags_by_service, out):
    config = config_text(model, "otelcol_app_config")
    if not config:
        return
    where = f"{path}: configs.otelcol_app_config"
    if "datadog-agent:4317" in config or "otlp/datadog_agent" in config:
        out.append(Finding(path, 0, "stack.otel-scrape",
                           f"{where}: standalone inference compose files must not export OTLP to the agent",
                           key="agent-export"))
    try:
        targets = scrape_targets(config)
    except YamlError:
        out.append(Finding(path, 0, "stack.otel-scrape", f"{where}: invalid collector config", key="parse"))
        return
    expected_ports = expected_public_ports(model, services)
    scrape_services = set()
    for name, svc in services.items():
        labels = svc.get("labels") or {}
        if _s(labels.get("nearai.otel.scrape")) != "true":
            continue
        scrape_services.add(name)
        for key in REQUIRED_OTEL_SCRAPE_LABELS:
            if _s(labels.get(key)) == "":
                out.append(Finding(path, 0, "stack.otel-scrape", f"{path}: services.{name}.labels: missing {key}",
                                   key=f"{name}:{key}"))
        target = targets.get(name)
        if target is None:
            out.append(Finding(path, 0, "stack.otel-scrape", f"{where}: missing scrape target for {name}",
                               key=f"{name}:target"))
            continue
        expected_target = f"{name}:{_s(labels.get('nearai.otel.port'))}"
        if _s(target["target"]) != expected_target:
            out.append(Finding(path, 0, "stack.otel-scrape",
                               f"{where}: {name} target {target['target']!r} should be {expected_target!r}",
                               key=f"{name}:target-value"))
        if target["metrics_path"] != labels.get("nearai.otel.path"):
            out.append(Finding(path, 0, "stack.otel-scrape",
                               f"{where}: {name} metrics_path {target['metrics_path']!r} does not match "
                               "nearai.otel.path", key=f"{name}:path"))
        tlabels = target["labels"]
        expected_labels = {
            "service": labels.get("nearai.otel.service"), "source": labels.get("nearai.otel.source"),
            "container_name": labels.get("nearai.otel.container_name") or name,
            "model": labels.get("nearai.otel.model"), "deployment": labels.get("nearai.otel.deployment"),
            "env": labels.get("nearai.otel.env"), "host": labels.get("nearai.otel.host"),
            "host_machine": labels.get("nearai.otel.host_machine"), "cvm_name": labels.get("nearai.otel.cvm_name"),
            "ip": labels.get("nearai.otel.ip"),
        }
        for key, expected in expected_labels.items():
            if _s(expected) == "" or tlabels.get(key) == expected:
                continue
            out.append(Finding(path, 0, "stack.otel-scrape",
                               f"{where}: {name} {key} label {tlabels.get(key)!r} should be {expected!r}",
                               key=f"{name}:{key}"))
        log_tags = tags_by_service.get(name) or {}
        for key in OPTIONAL_SCRAPE_TAGS:
            expected = log_tags.get(key)
            if expected is None or tlabels.get(key) == expected:
                continue
            out.append(Finding(path, 0, "stack.otel-scrape",
                               f"{where}: {name} optional {key} label {tlabels.get(key)!r} should be {expected!r}",
                               key=f"{name}:opt-{key}"))
        expected_port = expected_ports.get(name)
        if expected_port is None:
            expected_port = log_tags.get("port")
        if expected_port is None:
            expected_port = labels.get("nearai.otel.port")
        if expected_port is not None and tlabels.get("port") != expected_port:
            out.append(Finding(path, 0, "stack.otel-scrape",
                               f"{where}: {name} public port label {tlabels.get('port')!r} should be {expected_port!r}",
                               key=f"{name}:port"))
    for name in sorted(set(targets) - scrape_services):
        out.append(Finding(path, 0, "stack.otel-scrape",
                           f"{where}: extra scrape target {name} has no nearai.otel.scrape=true service",
                           key=f"{name}:extra"))


def otel_scope(services):
    """Files that run a stack whose logs and metrics we collect: a collector, an engine or a proxy."""
    return any(n == "otelcol-contrib" or is_proxy_name(n) or ENGINE_IMAGE.match(image_of(s))
               or PROXY_IMAGE.match(image_of(s)) for n, s in services.items())


# ---------------------------------------------------------------- guard

def check(ctx):
    out = []
    isolated = set(ctx.services_allow.get("isolated_files", []))
    deployables = ctx.head.deployables()
    for path in sorted(isolated - set(deployables)):
        out.append(Finding(path, 0, "stack.engine-network", f"{path}: listed in isolated_files but missing",
                           key=path))
    for path in deployables:
        model = ctx.raw(path)
        if model is None:
            continue
        services = model.get("services") or {}
        if path in isolated:
            pending = {e.get("service") for e in ctx.services_allow.get("services", [])
                       if e.get("file") == path and e.get("isolation") == "pending"}
            check_isolation(path, model, pending, out)
        check_proxy_depends(path, model, services, out)
        check_proxy_env(path, services, out)
        check_registrar(path, model, services, out)
        if otel_scope(services):
            check_otel_collector(path, model, services, out)
            tags = check_otel_logs(path, services, out)
            check_otel_scrape(path, model, services, tags, out)
    return out


# ---------------------------------------------------------------- self-test fixtures

ENGINE_IMG = "lmsysorg/sglang@sha256:" + "0" * 64
PROXY_IMG = "nearaidev/vllm-proxy-rs@sha256:" + "0" * 64
COLLECTOR_IMG = "otel/opentelemetry-collector-contrib@sha256:" + "0" * 64
SCRAPE_CFG = """receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 127.0.0.1:4317
  prometheus/apps:
    config:
      scrape_configs:
        - job_name: e
          metrics_path: /metrics
          static_configs:
            - targets: ['e:8000']
              labels:
                service: "sglang"
                source: "sglang"
                container_name: "e"
                model: "m"
                deployment: "d"
                env: "prod"
                host: "h"
                host_machine: "h"
                cvm_name: "c"
                ip: "ci-ip"
                port: "8000"
exporters:
  otlphttp/gateway:
    endpoint: ci-gateway
    sending_queue:
      num_consumers: 1
"""
LOG_E = '[{"source":"sglang","service":"sglang","tags":["model:m","deployment:d","env:prod","host:h","ip:ci-ip"]}]'
PROXY_ENV = ["VLLM_BASE_URL=http://e:8000", "MODEL_NAME=m", "CLOUD_API_URL=u",
             "WEB_CONTEXT_SEARCH_URL=${WEB_CONTEXT_SEARCH_URL}",
             "WEB_CONTEXT_SEARCH_API_KEY=${WEB_CONTEXT_SEARCH_API_KEY}",
             "VLLM_PROXY_INLINE_MEDIA=${VLLM_PROXY_INLINE_MEDIA:-1}",
             "VLLM_PROXY_REJECTED_CONTENT_PART_TYPES=${VLLM_PROXY_REJECTED_CONTENT_PART_TYPES:-video_url}"]
GOOD = {
    "networks": {"eng": {"internal": True}},
    "volumes": {"otelcol_app_storage": {}},
    "configs": {
        "otelcol_app_config": {"content": SCRAPE_CFG},
        "registrar_script": {"content": 'INFERENCE_TOKEN="$${PROXY_TOKEN}"\ncurl -s -H "Authorization: Bearer '
                                        '$$INFERENCE_TOKEN" http://x/v1/models\n'},
    },
    "services": {
        "e": {"image": ENGINE_IMG, "networks": ["eng"], "environment": ["HF_HUB_OFFLINE=1"],
              "labels": {"com.datadoghq.ad.logs": LOG_E, "nearai.otel.scrape": "true",
                         "nearai.otel.service": "sglang", "nearai.otel.source": "sglang",
                         "nearai.otel.container_name": "e", "nearai.otel.model": "m",
                         "nearai.otel.deployment": "d", "nearai.otel.env": "prod", "nearai.otel.host": "h",
                         "nearai.otel.host_machine": "h", "nearai.otel.cvm_name": "c",
                         "nearai.otel.ip": "ci-ip", "nearai.otel.port": "8000", "nearai.otel.path": "/metrics"}},
        "proxy-x": {"image": PROXY_IMG, "networks": ["default", "eng"], "environment": PROXY_ENV},
        "otelcol-contrib": {"image": COLLECTOR_IMG, "networks": ["default"],
                            "environment": ["MONITORING_INGEST_TOKEN=${MONITORING_INGEST_TOKEN:-}",
                                            "CVM_NAME=${CVM_NAME:-unknown}", "CVM_HOST=${CVM_HOST:-unknown}",
                                            "DD_HOSTNAME=${DD_HOSTNAME:-unknown}", "ENV=${ENV:-unknown}",
                                            "HOST_IP=${HOST_IP:-unknown}"],
                            "configs": [{"source": "otelcol_app_config"}],
                            "labels": {"com.datadoghq.ad.logs": '[{"source":"otelcol-contrib",'
                                       '"service":"otelcol-contrib","tags":["deployment:d","env:prod",'
                                       '"host:h","ip:ci-ip"]}]'}},
        "model-proxy-registrar": {"image": "curlimages/curl@sha256:" + "0" * 64, "networks": ["default"],
                                  "environment": ["PROXY_TOKEN=${PROXY_TOKEN}"]},
    },
}
P = "prod/a.yaml"
FILES = {P: "services: {}\n", SERVICES: json.dumps({"services": [], "env": [], "isolated_files": [P]})}


def _mut(*steps):
    """A copy of GOOD with each (path, value) step applied; a step whose value is None deletes the key."""
    m = copy.deepcopy(GOOD)
    for path, value in steps:
        node = m
        for k in path[:-1]:
            node = node[k]
        if value is None:
            del node[path[-1]]
        else:
            node[path[-1]] = value
    return m


def _case(name, rule, model, files=FILES):
    return (name, rule, files, None, None, None, {"raw": {P: model}})


OTEL_CFG = SCRAPE_CFG
PENDING = dict(FILES, **{SERVICES: json.dumps({"isolated_files": [P], "services": [
    {"file": P, "service": "e", "isolation": "pending"}]})})
PENDING_GONE = dict(FILES, **{SERVICES: json.dumps({"isolated_files": [P], "services": [
    {"file": P, "service": "gone", "isolation": "pending"}]})})
NOT_ISOLATED = dict(FILES, **{SERVICES: json.dumps({"isolated_files": [], "services": [], "env": []})})
CASES = [
    ("compliant model", None, FILES, None, None, None, {"raw": {P: GOOD}}),
    ("engine outside isolated_files is not checked", None, NOT_ISOLATED, None, None, None,
     {"raw": {P: _mut((("networks", "eng", "internal"), False))}}),
    _case("engine network not internal", "stack.engine-network",
          _mut((("networks", "eng", "internal"), False))),
    _case("engine publishes a port", "stack.engine-network",
          _mut((("services", "e", "ports"), ["8000:8000"]))),
    _case("engine reads HF_HUB_OFFLINE from a variable", "stack.engine-network",
          _mut((("services", "e", "environment"), ["HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}"]))),
    _case("non-engine on an engine network", "stack.engine-member",
          _mut((("services", "x"), {"image": "nginx@sha256:" + "0" * 64, "networks": ["eng"]}))),
    _case("collector OTLP receiver not on loopback", "stack.engine-member",
          _mut((("services", "c"), {"image": COLLECTOR_IMG, "networks": ["eng"], "configs": [{"source": "col"}]}),
               (("configs", "col"), {"content": "receivers:\n  otlp:\n    protocols:\n      grpc:\n"
                                                "        endpoint: 0.0.0.0:4317\n"}))),
    _case("proxy backend names no service", "stack.engine-backend",
          _mut((("services", "proxy-x", "environment"),
                [v.replace("http://e:8000", "http://nowhere:8000") for v in PROXY_ENV]))),
    ("pending engine follows the contract", "stack.engine-pending", PENDING, None, None, None,
     {"raw": {P: GOOD}}),
    ("pending service is not in the file", "stack.engine-pending", PENDING_GONE, None, None, None,
     {"raw": {P: GOOD}}),
    _case("proxy depends on a model", "stack.proxy-depends",
          _mut((("services", "proxy-x", "depends_on"), {"model-x": {"condition": "service_started"}}))),
    _case("proxy env without the search key", "stack.proxy-env",
          _mut((("services", "proxy-x", "environment"),
                [v for v in PROXY_ENV if "WEB_CONTEXT_SEARCH_API_KEY" not in v]))),
    _case("registrar probe without the bearer header", "stack.registrar-auth",
          _mut((("configs", "registrar_script"),
                {"content": 'INFERENCE_TOKEN="$${PROXY_TOKEN}"\ncurl -s http://x/v1/models\n'}))),
    _case("collector marker missing for one variable", "stack.otel-collector",
          _mut((("services", "otelcol-contrib", "environment"),
                ["MONITORING_INGEST_TOKEN=${MONITORING_INGEST_TOKEN:-}", "CVM_NAME=c",
                 "CVM_HOST=${CVM_HOST:-unknown}", "DD_HOSTNAME=${DD_HOSTNAME:-unknown}",
                 "ENV=${ENV:-unknown}", "HOST_IP=${HOST_IP:-unknown}"]))),
    _case("gateway with two consumers", "stack.otel-collector",
          _mut((("configs", "otelcol_app_config", "content"),
                OTEL_CFG.replace("num_consumers: 1", "num_consumers: 2")))),
    _case("log label is not JSON", "stack.otel-logs",
          _mut((("services", "e", "labels", "com.datadoghq.ad.logs"), "[not json"))),
    _case("scrape label differs from the target", "stack.otel-scrape",
          _mut((("configs", "otelcol_app_config", "content"),
                OTEL_CFG.replace('host: "h"', 'host: "other"')))),
]
