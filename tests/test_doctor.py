"""
Tests for the agentci doctor command.
"""
import os
from click.testing import CliRunner
from ciagent.cli import cli


class TestDoctorCommand:
    """Tests for agentci doctor."""

    def test_doctor_no_spec(self, tmp_path):
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(cli, ["doctor"])
            assert result.exit_code == 1  # fail because no spec
            assert "not found" in result.output

    def test_doctor_with_valid_spec(self, tmp_path):
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            with open("agentci_spec.yaml", "w") as f:
                f.write(
                    "version: 1.0\nagent: test\nqueries:\n  - query: hello\n"
                )
            result = runner.invoke(cli, ["doctor"])
            assert "CIAgent Doctor" in result.output
            assert "valid" in result.output

    def test_doctor_with_invalid_spec(self, tmp_path):
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            with open("agentci_spec.yaml", "w") as f:
                f.write("not: valid: yaml: [")
            result = runner.invoke(cli, ["doctor"])
            assert result.exit_code == 1

    def test_doctor_checks_python_version(self, tmp_path):
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            with open("agentci_spec.yaml", "w") as f:
                f.write(
                    "version: 1.0\nagent: test\nqueries:\n  - query: hello\n"
                )
            result = runner.invoke(cli, ["doctor"])
            assert "Python" in result.output

    def test_doctor_checks_dependencies(self, tmp_path):
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            with open("agentci_spec.yaml", "w") as f:
                f.write(
                    "version: 1.0\nagent: test\nqueries:\n  - query: hello\n"
                )
            result = runner.invoke(cli, ["doctor"])
            # pydantic and click are always installed in test env
            assert "pydantic" in result.output
            assert "click" in result.output

    def test_doctor_shows_summary(self, tmp_path):
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            with open("agentci_spec.yaml", "w") as f:
                f.write(
                    "version: 1.0\nagent: test\nqueries:\n  - query: hello\n"
                )
            result = runner.invoke(cli, ["doctor"])
            assert "passed" in result.output
            assert "warnings" in result.output
            assert "failures" in result.output
