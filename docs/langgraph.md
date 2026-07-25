# LangGraph Integration

CIAgent has first-class support for LangGraph.

## setup

1. Install the optional dependency:
   ```bash
   pip install "ciagent[langgraph]"
   ```

2. Configure your `ciagent.yaml`:
   ```yaml
   framework: "langgraph"
   agent: "my_graph:app"
   ```

The CIAgent adapter will automatically instrument your LangGraph application to capture traces.
