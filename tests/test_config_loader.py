# Copyright 2025-2026 The CIAgent Authors
# SPDX-License-Identifier: Apache-2.0
"""
Regression tests for the v1 suite loader (ciagent.loader.load_suite) and the
deprecated ciagent.config.load_config shim.

Guards against the silent-failure class where the v1 loader, handed a v2 spec
(ciagent_spec.yaml with a `queries:` key, as written by `ciagent init
--generate`), returned a TestSuite with zero tests and no warning.
"""

import pytest

from ciagent.exceptions import ConfigError
from ciagent.loader import detect_format, load_suite as load_config


V1_SUITE = """\
name: my-suite
agent: myapp.agent:run_agent
tests:
  - name: test_billing
    input: "I have a billing question"
    golden_trace: golden/billing.json
"""

# Mirrors the shape written by `ciagent init --generate` (cli.py skeleton spec).
V2_SPEC = """\
agent: my-agent
runner: "myagent.run:run_agent"
version: 1.0
baseline_dir: ./baselines
queries:
  - query: "What is your refund policy?"
    description: "In-scope test"
    cost:
      max_llm_calls: 10
"""


def _write(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return str(path)


class TestLoadConfigV1:
    def test_loads_valid_v1_suite(self, tmp_path):
        suite = load_config(_write(tmp_path, "ciagent.yaml", V1_SUITE))
        assert suite.name == "my-suite"
        assert len(suite.tests) == 1
        assert suite.tests[0].name == "test_billing"

    def test_resolves_golden_trace_relative_to_config(self, tmp_path):
        suite = load_config(_write(tmp_path, "ciagent.yaml", V1_SUITE))
        assert suite.tests[0].golden_trace == str(tmp_path / "golden/billing.json")

    def test_missing_file_raises_config_error(self, tmp_path):
        with pytest.raises(ConfigError, match="not found"):
            load_config(str(tmp_path / "nope.yaml"))


class TestLoadConfigRejectsV2Spec:
    """load_config must never load a generated v2 spec as zero tests."""

    def test_generated_spec_raises_instead_of_zero_tests(self, tmp_path):
        path = _write(tmp_path, "ciagent_spec.yaml", V2_SPEC)
        with pytest.raises(ConfigError, match="v2 spec"):
            load_config(path)

    def test_error_points_to_v2_loader(self, tmp_path):
        path = _write(tmp_path, "ciagent_spec.yaml", V2_SPEC)
        with pytest.raises(ConfigError, match="ciagent test|load_spec"):
            load_config(path)

    def test_queries_key_alone_is_enough_to_reject(self, tmp_path):
        path = _write(tmp_path, "spec.yaml", "agent: a\nqueries: []\n")
        with pytest.raises(ConfigError, match="queries"):
            load_config(path)


class TestLoadConfigUnknownKeys:
    def test_unknown_keys_without_tests_raise(self, tmp_path):
        path = _write(tmp_path, "ciagent.yaml", "agent: a\ntest_cases:\n  - x\n")
        with pytest.raises(ConfigError, match="test_cases"):
            load_config(path)

    def test_unknown_keys_with_tests_warn_but_load(self, tmp_path):
        path = _write(tmp_path, "ciagent.yaml", V1_SUITE + "extra_key: 1\n")
        with pytest.warns(UserWarning, match="extra_key"):
            suite = load_config(path)
        assert len(suite.tests) == 1


class TestDetectFormat:
    def test_v2_spec_detected(self, tmp_path):
        assert detect_format(_write(tmp_path, "s.yaml", V2_SPEC)) == "v2"

    def test_v1_suite_detected(self, tmp_path):
        assert detect_format(_write(tmp_path, "s.yaml", V1_SUITE)) == "v1"

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ConfigError, match="not found"):
            detect_format(str(tmp_path / "nope.yaml"))

    def test_unclassifiable_file_raises(self, tmp_path):
        path = _write(tmp_path, "s.yaml", "agent: a\nname: x\n")
        with pytest.raises(ConfigError, match="neither"):
            detect_format(path)


class TestDeprecatedConfigShim:
    def test_load_config_warns_and_delegates(self, tmp_path):
        from ciagent.config import load_config as shim
        path = _write(tmp_path, "ciagent.yaml", V1_SUITE)
        with pytest.warns(DeprecationWarning, match="load_suite"):
            suite = shim(path)
        assert len(suite.tests) == 1

    def test_shim_still_rejects_v2_specs(self, tmp_path):
        from ciagent.config import load_config as shim
        path = _write(tmp_path, "ciagent_spec.yaml", V2_SPEC)
        with pytest.warns(DeprecationWarning):
            with pytest.raises(ConfigError, match="v2 spec"):
                shim(path)


class TestLoadConfigMalformedYaml:
    def test_non_mapping_yaml_raises(self, tmp_path):
        path = _write(tmp_path, "ciagent.yaml", "- just\n- a list\n")
        with pytest.raises(ConfigError, match="mapping"):
            load_config(path)

    def test_empty_file_raises(self, tmp_path):
        path = _write(tmp_path, "ciagent.yaml", "")
        with pytest.raises(ConfigError, match="mapping"):
            load_config(path)

    def test_yaml_syntax_error_raises_config_error(self, tmp_path):
        path = _write(tmp_path, "ciagent.yaml", "tests: [unclosed\n")
        with pytest.raises(ConfigError, match="Invalid YAML"):
            load_config(path)
