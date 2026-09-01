#!/usr/bin/env ruby
# Protect the production GLM-5.3 H200 hardware-telemetry contract.

require "yaml"

ROOT = File.expand_path("..", __dir__)
COMPOSE_FILE = File.join(ROOT, "prod", "GLM-5.3-Flash-SGL-TP4.yaml")
EXPECTED_IMAGE = "nvcr.io/nvidia/k8s/dcgm-exporter@sha256:613ab03c11d442fd960ff515f547e9921537454a712d08160bc8f677f89f1c35"
COLLECTOR_SOURCE = "dcgm_h200_metrics"
COLLECTOR_PATH = "/etc/dcgm-exporter/nearai-h200.csv"
REQUIRED_FIELDS = %w[
  DCGM_FI_DEV_SM_CLOCK
  DCGM_FI_DEV_MEM_CLOCK
  DCGM_FI_DEV_GPU_UTIL
  DCGM_FI_DEV_MEM_COPY_UTIL
  DCGM_FI_DEV_FB_FREE
  DCGM_FI_DEV_FB_USED
  DCGM_FI_DEV_FB_RESERVED
  DCGM_FI_DEV_GPU_TEMP
  DCGM_FI_DEV_MEMORY_TEMP
  DCGM_FI_DEV_POWER_USAGE
  DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION
  DCGM_FI_PROF_PCIE_TX_BYTES
  DCGM_FI_PROF_PCIE_RX_BYTES
  DCGM_FI_DEV_PCIE_REPLAY_COUNTER
  DCGM_FI_DEV_XID_ERRORS
  DCGM_EXP_XID_ERRORS_COUNT
  DCGM_EXP_XID_ERRORS_TOTAL
  DCGM_EXP_CLOCK_EVENTS_COUNT
  DCGM_EXP_CLOCK_EVENTS_TOTAL
  DCGM_EXP_GPU_HEALTH_STATUS
  DCGM_EXP_P2P_STATUS
  DCGM_FI_DEV_POWER_VIOLATION
  DCGM_FI_DEV_THERMAL_VIOLATION
  DCGM_FI_DEV_SYNC_BOOST_VIOLATION
  DCGM_FI_DEV_BOARD_LIMIT_VIOLATION
  DCGM_FI_DEV_LOW_UTIL_VIOLATION
  DCGM_FI_DEV_RELIABILITY_VIOLATION
  DCGM_FI_DEV_ECC_SBE_VOL_TOTAL
  DCGM_FI_DEV_ECC_DBE_VOL_TOTAL
  DCGM_FI_DEV_ECC_SBE_AGG_TOTAL
  DCGM_FI_DEV_ECC_DBE_AGG_TOTAL
  DCGM_FI_DEV_RETIRED_SBE
  DCGM_FI_DEV_RETIRED_DBE
  DCGM_FI_DEV_RETIRED_PENDING
  DCGM_FI_DEV_UNCORRECTABLE_REMAPPED_ROWS
  DCGM_FI_DEV_CORRECTABLE_REMAPPED_ROWS
  DCGM_FI_DEV_ROW_REMAP_FAILURE
  DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL
  DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL
  DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_TOTAL
  DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_TOTAL
  DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_TOTAL
  DCGM_FI_PROF_GR_ENGINE_ACTIVE
  DCGM_FI_PROF_PIPE_TENSOR_ACTIVE
  DCGM_FI_PROF_DRAM_ACTIVE
  DCGM_FI_DRIVER_VERSION
  DCGM_FI_NVML_VERSION
  DCGM_FI_DEV_BRAND
  DCGM_FI_DEV_VBIOS_VERSION
].freeze

def yaml_load(content)
  YAML.load(content, aliases: true)
rescue ArgumentError
  YAML.load(content)
end

def config_source(config)
  config.is_a?(String) ? config : config["source"]
end

def active_fields(content)
  content.lines.filter_map do |line|
    stripped = line.strip
    next if stripped.empty? || stripped.start_with?("#")

    stripped.split(",", 2).first.strip
  end
end

errors = []
compose = yaml_load(File.read(COMPOSE_FILE))
service = compose.dig("services", "dcgm-glm53")

if service.nil?
  errors << "missing services.dcgm-glm53"
else
  errors << "dcgm-glm53 image must be #{EXPECTED_IMAGE}" unless service["image"] == EXPECTED_IMAGE

  command = Array(service["command"])
  unless command == ["-f", COLLECTOR_PATH]
    errors << "dcgm-glm53 command must load #{COLLECTOR_PATH} with -f"
  end

  collector_mount = Array(service["configs"]).find do |config|
    config_source(config) == COLLECTOR_SOURCE
  end
  if collector_mount.nil?
    errors << "dcgm-glm53 must mount configs.#{COLLECTOR_SOURCE}"
  elsif !collector_mount.is_a?(Hash)
    errors << "dcgm-glm53 collector mount must use long syntax with an explicit target"
  else
    errors << "dcgm-glm53 collector target must be #{COLLECTOR_PATH}" unless collector_mount["target"] == COLLECTOR_PATH
    errors << "dcgm-glm53 collector mount must be read-only mode 0444" unless collector_mount["mode"] == 0o444
  end

  device_ids = service.dig("deploy", "resources", "reservations", "devices", 0, "device_ids")
  errors << "dcgm-glm53 must cover GPU device_ids 0-7" unless device_ids == (0..7).map(&:to_s)
end

collector = compose.dig("configs", COLLECTOR_SOURCE, "content").to_s
if collector.empty?
  errors << "missing configs.#{COLLECTOR_SOURCE}.content"
else
  fields = active_fields(collector)
  duplicates = fields.tally.select { |_field, count| count > 1 }.keys
  errors << "duplicate active DCGM fields: #{duplicates.join(', ')}" unless duplicates.empty?

  missing = REQUIRED_FIELDS - fields
  errors << "missing active DCGM fields: #{missing.join(', ')}" unless missing.empty?
end

otel_content = compose.dig("configs", "otelcol_app_config", "content").to_s
otel = yaml_load(otel_content)
dcgm_scrape = Array(otel.dig("receivers", "prometheus/apps", "config", "scrape_configs")).find do |scrape|
  scrape["job_name"] == "dcgm-dcgm-glm53"
end
if dcgm_scrape.nil?
  errors << "missing dcgm-dcgm-glm53 scrape job"
else
  errors << "dcgm scrape_interval must be 15s" unless dcgm_scrape["scrape_interval"] == "15s"
  errors << "dcgm scrape_timeout must be 10s" unless dcgm_scrape["scrape_timeout"] == "10s"
end

if errors.any?
  warn "GLM-5.3 DCGM telemetry contract failed:"
  errors.each { |error| warn "  - #{error}" }
  exit 1
end

puts "GLM-5.3 DCGM telemetry contract OK"
