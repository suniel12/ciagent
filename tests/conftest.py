# Copyright 2025-2026 The CIAgent Authors
# SPDX-License-Identifier: Apache-2.0
"""Keep the legacy-fallback fixture out of our own suite's collection.

The pytest plugin (correctly) collects legacy-named spec files; the copy
under fixtures/legacy exists to test the deprecation fallbacks, not to run
as a live spec against a nonexistent agent.
"""

collect_ignore_glob = ["fixtures/legacy/*"]
