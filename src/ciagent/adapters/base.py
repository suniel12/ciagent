# Copyright 2025-2026 The AgentCI Authors
# SPDX-License-Identifier: Apache-2.0
"""
Base Adapter class.
"""

from typing import Any, Sequence

class BaseAdapter:
    def run(self, agent: Any, input_data: Any) -> Any:
        raise NotImplementedError
        
    @staticmethod
    def extract_tool_schemas(tools: Sequence[Any]) -> list[dict[str, Any]]:
        """
        Utility to automatically extract JSON schemas from a list of tools or Pydantic models.
        This prevents 'Object of type ModelMetaclass is not JSON serializable' errors
        when passing tool definitions to LLM libraries like Anthropic or OpenAI.
        """
        schemas = []
        for tool in tools:
            # If it's already a dict (e.g., standard OpenAI format), keep it
            if isinstance(tool, dict):
                schemas.append(tool)
                continue
                
            # If it's a Pydantic V2 model class (has model_json_schema)
            if hasattr(tool, "model_json_schema") and callable(tool.model_json_schema):
                schemas.append(tool.model_json_schema())
                continue
                
            # If it's a Pydantic V1 model class (has schema)
            if hasattr(tool, "schema") and callable(tool.schema):
                schemas.append(tool.schema())
                continue
                
            # If the tool object itself has an input_schema property that is a model
            if hasattr(tool, "input_schema"):
                schema_attr = getattr(tool, "input_schema")
                if isinstance(schema_attr, dict):
                    schemas.append(schema_attr)
                elif hasattr(schema_attr, "model_json_schema") and callable(schema_attr.model_json_schema):
                    schemas.append(schema_attr.model_json_schema())
                elif hasattr(schema_attr, "schema") and callable(schema_attr.schema):
                    schemas.append(schema_attr.schema())
                else:
                    schemas.append(tool)
            else:
                schemas.append(tool)
                
        return schemas
