#!/usr/bin/env ruby
require "fileutils"
require "open3"
require "tmpdir"

ROOT = File.expand_path("..", __dir__)
TARGET = "prod/GLM-5.1-SGL-AWQ-TP4.yaml"
OPT_OUT_TARGET = "prod/GLM-5.2-SGL-FP8-TP8.yaml"

def copy_fixture(destination)
  %w[prod experiments].each do |directory|
    FileUtils.cp_r(File.join(ROOT, directory), destination)
  end
  Dir.glob(File.join(ROOT, "*.yaml")).each do |path|
    FileUtils.cp(path, destination)
  end
  FileUtils.mkdir_p(File.join(destination, "scripts"))
  FileUtils.cp(File.join(ROOT, "scripts", "validate_otel_labels.rb"), File.join(destination, "scripts"))
end

def run_validator(root)
  Open3.capture3("ruby", "scripts/validate_otel_labels.rb", chdir: root)
end

def assert_changed(name, original, mutated)
  abort("#{name}: fixture mutation did not match") if original == mutated
end

def expect_failure(root, name, target, content, expected)
  File.write(target, content)
  stdout, stderr, status = run_validator(root)
  output = stdout + stderr
  abort("#{name}: validator unexpectedly passed") if status.success?
  abort("#{name}: expected #{expected.inspect}, got #{output.inspect}") unless output.include?(expected)
end

Dir.mktmpdir("otel-label-contract-") do |fixture|
  copy_fixture(fixture)
  target = File.join(fixture, TARGET)
  baseline = File.read(target)

  stdout, stderr, status = run_validator(fixture)
  abort("baseline failed: #{stdout}#{stderr}") unless status.success?

  missing_pair = baseline.sub(
    /^(\s*)nearai\.otel\.logs:.*\n\1com\.datadoghq\.ad\.logs:.*\n/,
    "",
  )
  assert_changed("missing pair", baseline, missing_pair)
  expect_failure(fixture, "missing pair", target, missing_pair, "missing neutral log metadata label")

  missing_datadog = baseline.sub(/^\s*com\.datadoghq\.ad\.logs:.*\n/, "")
  assert_changed("missing Datadog parity", baseline, missing_datadog)
  expect_failure(fixture, "missing Datadog parity", target, missing_datadog, "missing temporary Datadog parity label")

  divergent_pair = baseline.sub(
    /(^\s*nearai\.otel\.logs: '[^'\n]*"source":")[^"]+/,
    '\1validator-test',
  )
  assert_changed("divergent pair", baseline, divergent_pair)
  expect_failure(fixture, "divergent pair", target, divergent_pair, "does not match temporary Datadog parity label")

  datadog_collector = baseline.sub(
    'attributes["attrs"]["nearai.otel.logs"]',
    'attributes["attrs"]["com.datadoghq.ad.logs"]',
  )
  assert_changed("Datadog collector dependency", baseline, datadog_collector)
  expect_failure(fixture, "Datadog collector dependency", target, datadog_collector, "must not depend on Datadog-specific metadata")

  renamed_filelog = baseline
                    .gsub("filelog/docker_containers", "filelog/apps")
                    .gsub("nearai.otel.logs", "unrelated.logs")
  assert_changed("renamed filelog receiver", baseline, renamed_filelog)
  expect_failure(fixture, "renamed filelog receiver", target, renamed_filelog, "Docker log pipeline does not consume nearai.otel.logs")

  multiple_entries = baseline.sub(/^\s*nearai\.otel\.logs:.*$/) do |line|
    line.sub(
      /\}\]'$/,
      '},{"source":"validator-test","service":"validator-test","tags":[]}]\'',
    )
  end
  assert_changed("multiple entries", baseline, multiple_entries)
  expect_failure(fixture, "multiple entries", target, multiple_entries, "expected a single-entry JSON array")

  opt_out_target = File.join(fixture, OPT_OUT_TARGET)
  opt_out_baseline = File.read(opt_out_target)
  missing_opt_out = opt_out_baseline.sub(/^\s*nearai\.otel\.logs\.disabled: "true"\n/, "")
  assert_changed("missing opt-out", opt_out_baseline, missing_opt_out)
  expect_failure(fixture, "missing opt-out", opt_out_target, missing_opt_out, "approved log collection opt-out must remain true")

  invalid_opt_out = opt_out_baseline.sub('nearai.otel.logs.disabled: "true"', 'nearai.otel.logs.disabled: "false"')
  assert_changed("invalid opt-out", opt_out_baseline, invalid_opt_out)
  expect_failure(fixture, "invalid opt-out", opt_out_target, invalid_opt_out, "approved log collection opt-out must remain true")

  unapproved_opt_out = missing_pair.sub(/^(\s*)labels:\n/) do
    indentation = Regexp.last_match(1)
    "#{indentation}labels:\n#{indentation}  nearai.otel.logs.disabled: \"true\"\n"
  end
  assert_changed("unapproved opt-out", missing_pair, unapproved_opt_out)
  expect_failure(fixture, "unapproved opt-out", target, unapproved_opt_out, "log collection opt-out is not approved for this service")

  File.write(target, baseline)
  File.write(opt_out_target, opt_out_baseline)
  nested = File.join(fixture, "experiments", "nested", "missing-labels.yaml")
  FileUtils.mkdir_p(File.dirname(nested))
  File.write(nested, missing_pair)
  _stdout, nested_stderr, nested_status = run_validator(fixture)
  abort("nested discovery: validator unexpectedly passed") if nested_status.success?
  abort("nested discovery: missing nested path") unless nested_stderr.include?("experiments/nested/missing-labels.yaml")
end

puts "OTel label contract regression tests OK"
