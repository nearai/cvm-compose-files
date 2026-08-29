#!/usr/bin/env ruby
# frozen_string_literal: true

require "yaml"

root = File.expand_path("..", __dir__)
relative = "experiments/DeepSeek-V4-Flash-SGL-FP4-4xTP2-bare-test.yaml"
path = File.join(root, relative)
compose = YAML.load(File.read(path), aliases: true)
services = compose.fetch("services")
errors = []

expected_services = %w[
  model-downloader
  model-sg-dsv4-flash-fp4-tp2-r1
  model-sg-dsv4-flash-fp4-tp2-r2
  model-sg-dsv4-flash-fp4-tp2-r3
  model-sg-dsv4-flash-fp4-tp2-r4
  dcgm-dsv4-flash
  otelcol-contrib
]
expected_engine_image = "lmsysorg/sglang@sha256:6bb5fee34b6c4537c09a4775e2292ac40350d5ad1218fcc835b2692142f443b1"

engines = services.select { |name, _| name.match?(/^model-sg-dsv4-flash-fp4-tp2-r[1-4]$/) }
errors << "expected exactly four TP2 engine services" unless engines.length == 4
errors << "service set mismatch" unless services.keys.sort == expected_services.sort

forbidden_services = services.keys.grep(/proxy|registrar|nginx|dstack|vmm/)
errors << "forbidden services: #{forbidden_services.join(", ")}" unless forbidden_services.empty?

expected_ports = %w[
  127.0.0.1:30001:8000
  127.0.0.1:30002:8000
  127.0.0.1:30003:8000
  127.0.0.1:30004:8000
]

engines.sort.each_with_index do |(name, service), offset|
  index = offset + 1
  errors << "#{name} container_name mismatch" unless service["container_name"] == name
  errors << "#{name} image mismatch" unless service["image"] == expected_engine_image
  errors << "#{name} must bind one loopback port" unless service["ports"] == [expected_ports[offset]]
  errors << "#{name} must not set runtime" if service.key?("runtime")
  errors << "#{name} must carry a CPU set" unless service["cpuset"] == "${GPU_R#{index}_CPUSET:?GPU_R#{index}_CPUSET is required}"

  command = Array(service["command"]).join("\n")
  errors << "#{name} missing exact model revision" unless command.include?("553034d7dd9e06c2eeaee68cf85a17d6d4754cf0")
  errors << "#{name} missing TP2" unless command.include?("--tp 2")
  errors << "#{name} missing mixed chunking" unless command.include?("--enable-mixed-chunk")
  errors << "#{name} missing 8192 prefill budget" unless command.include?("--chunked-prefill-size 8192")
  errors << "#{name} enables speculative decoding" if command.match?(/speculative|EAGLE/i)

  environment = Array(service["environment"]).join("\n")
  errors << "#{name} enables Spec V2" if environment.include?("SGLANG_ENABLE_SPEC_V2")
  errors << "#{name} missing NUMA binding" unless environment.include?("GPU_NUMA_NODE=${GPU_R#{index}_MEMS:")

  devices = service.dig("deploy", "resources", "reservations", "devices")
  device_ids = Array(devices).first&.fetch("device_ids", nil)
  expected_ids = [
    "${GPU_R#{index}_0:?GPU_R#{index}_0 is required}",
    "${GPU_R#{index}_1:?GPU_R#{index}_1 is required}",
  ]
  errors << "#{name} device UUID contract mismatch" unless device_ids == expected_ids
end

services.each do |name, service|
  errors << "#{name} container_name mismatch" unless service["container_name"] == name
  errors << "#{name} must be digest pinned" unless service["image"].to_s.match?(/@sha256:[0-9a-f]{64}\z/)
  Array(service["ports"]).each do |port|
    errors << "#{name} has a non-loopback port #{port}" unless port.start_with?("127.0.0.1:")
  end
  errors << "#{name} is privileged" if service["privileged"]
end

file_text = File.read(path)
%w[/var/run/dstack.sock model-proxy-registrar SGLANG_ENABLE_SPEC_V2].each do |forbidden|
  errors << "contains forbidden token #{forbidden}" if file_text.include?(forbidden)
end

if errors.any?
  errors.each { |error| warn "::error file=#{relative}::#{error}" }
  exit 1
end

puts "gpu31 bare-metal experiment contract OK"
