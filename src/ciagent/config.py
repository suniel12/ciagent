# Copyright 2025-2026 The AgentCI Authors
# SPDX-License-Identifier: Apache-2.0
"""
Configuration loader.

Loads and validates agentci.yaml configuration files.
"""

from .models import TestSuite

import yaml
import os
from .models import TestSuite

def load_config(path: str = "agentci.yaml") -> TestSuite:
    """Load and validate agentci.yaml configuration file.

    Args:
        path: Path to the YAML config file (default: agentci.yaml).

    Returns:
        A validated TestSuite object.

    Raises:
        ConfigError: If the config file is missing or invalid.
    """
    if not os.path.exists(path):
        from .exceptions import ConfigError
        raise ConfigError(
            f"Configuration file not found: {path}",
            fix=f"Run 'ciagent init' to generate a default agentci.yaml, "
                f"or create one manually. See AGENTS.md for the expected format."
        )

    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    suite = TestSuite(**config_dict)
    
    # Resolve relative paths relative to the config file
    base_dir = os.path.dirname(os.path.abspath(path))
    
    for test in suite.tests:
        if test.golden_trace and not os.path.isabs(test.golden_trace):
            test.golden_trace = os.path.join(base_dir, test.golden_trace)
            
    return suite
