# AgentCI × Support Router: Incremental Build Plan

## The Philosophy (Same as RAG Demo)

Build the agent one step at a time. At each step, wrap it with AgentCI, look at the trace, ask "what could go wrong here?", write the test, intentionally break it, confirm the test catches it. Fix AgentCI when it falls short.

## Why This Demo Matters Strategically

| What It Proves | Why It Matters |
|----------------|---------------|
| AgentCI works with OpenAI Agents SDK | The largest single developer audience |
| AgentCI traces multi-agent handoffs | Nobody else does this well |
| Same assertion primitives work across frameworks | `assert trace.tools_called`, `assert trace.path` work for LangGraph AND OpenAI SDK |
| AgentCI captures routing decisions | The #1 failure mode in production support bots |

## The Critical Architectural Insight

The OpenAI Agents SDK has a **built-in tracing system** with a pluggable `TracingProcessor` interface:

```python
from agents.tracing import add_trace_processor

class AgentCITraceProcessor:
    """Captures OpenAI Agents SDK traces into AgentCI's data model."""
    
    def on_trace_start(self, trace): ...
    def on_span_start(self, span): ...
    def on_span_end(self, span): ...
    def on_trace_end(self, trace): ...
    def shutdown(self): ...
    def force_flush(self): ...

# Register AgentCI as a trace processor
add_trace_processor(AgentCITraceProcessor())
```

The SDK emits typed spans:
- `AgentSpanData` — which agent ran, its tools and handoffs
- `GenerationSpanData` — LLM input/output, model, token usage
- `HandoffSpanData` — from_agent, to_agent (THIS IS GOLD)
- `GuardrailSpanData` — name, triggered (true/false)
- `FunctionSpanData` — tool name, input, output

**This means AgentCI doesn't need to monkey-patch anything.** It implements a standard interface that the SDK already supports. Compare this to LangGraph where you had to build `ctx.attach_langgraph_state()`. The OpenAI path is architecturally cleaner.

This `AgentCITraceProcessor` becomes a first-class feature of the AgentCI library — any developer using the OpenAI Agents SDK gets trace capture with two lines of code.

---

## The Domain: TechCorp Customer Support

Similar to the NovaCorp knowledge base for the RAG demo, you need a controlled, fictional context. **TechCorp** is a SaaS company with:

**Products:** CloudSync Pro ($49/mo), CloudSync Business ($199/mo), CloudSync Enterprise ($499/mo)

**Support categories and their specialist agents:**

| Agent | Handles | Tools Available |
|-------|---------|----------------|
| Triage Agent | Classifies intent, routes to specialist | `handoff(billing)`, `handoff(technical)`, `handoff(account)`, `handoff(general)` |
| Billing Agent | Charges, invoices, plan changes, refunds | `lookup_invoice`, `process_refund`, `change_plan` |
| Technical Agent | Bugs, outages, integration help | `check_system_status`, `lookup_error_code`, `create_ticket` |
| Account Agent | Password reset, 2FA, profile changes, cancellation | `reset_password`, `toggle_2fa`, `cancel_account` |
| General Agent | FAQ, feature requests, anything else | `search_faq` |

**Why 4 specialists (not 2):** With 2 agents (billing/refund), the routing is trivially easy. With 4 agents, the LLM has to make genuinely ambiguous decisions. "I can't log in and I'm being charged" — is that Technical or Billing or Account? That ambiguity is where the interesting test cases emerge.

---

## Step 0: Scaffold (30 minutes)

### What you build

```
examples/support-router/
├── README.md
├── pyproject.toml
├── requirements.txt              # openai-agents, pydantic, agentci
├── .env.example                  # OPENAI_API_KEY
│
├── support_router/
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── triage.py             # Triage agent definition
│   │   ├── billing.py            # Billing specialist
│   │   ├── technical.py          # Technical specialist
│   │   ├── account.py            # Account specialist
│   │   └── general.py            # General/FAQ specialist
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── billing_tools.py      # lookup_invoice, process_refund, change_plan
│   │   ├── technical_tools.py    # check_system_status, lookup_error_code, create_ticket
│   │   ├── account_tools.py      # reset_password, toggle_2fa, cancel_account
│   │   └── general_tools.py      # search_faq
│   ├── context.py                # Shared customer context (UserContext dataclass)
│   └── run.py                    # CLI runner
│
├── tests/
│   ├── conftest.py               # AgentCI fixtures, mock setup
│   ├── fixtures/
│   │   ├── customers.py          # Test customer profiles
│   │   └── mock_responses.yaml   # Mocked LLM and tool responses
│   ├── golden_traces/            # Baseline traces
│   ├── test_routing.py           # Routing correctness tests
│   ├── test_tool_calls.py        # Tool selection within specialists
│   ├── test_guardrails.py        # Cost, boundary, and safety tests
│   ├── test_handoffs.py          # Handoff-specific assertions
│   └── test_golden_traces.py     # Regression tests
│
└── Makefile                      # make test, make test-live, make record-golden
```

### Commit message
`feat(examples): scaffold Support Router project with TechCorp domain`

---

## Step 1: The Simplest Triage — Two Agents (1-2 hours)

### What you build
The absolute minimum from the OpenAI Agents SDK:
- A Triage Agent with instructions to classify customer intent
- TWO specialist agents (Billing and General) — just two, not four yet
- Handoffs defined on the Triage Agent
- No tools on the specialists yet — they just respond with their instructions
- The `AgentCITraceProcessor` registered to capture traces

```python
from agents import Agent, Runner, handoff

billing_agent = Agent(
    name="Billing Agent",
    instructions="You handle billing questions for TechCorp. Be helpful and concise.",
    handoff_description="Customer has a billing, invoice, charge, or refund question",
)

general_agent = Agent(
    name="General Agent",
    instructions="You handle general questions about TechCorp products and services.",
    handoff_description="Customer has a general question, feature request, or FAQ",
)

triage_agent = Agent(
    name="Triage Agent",
    instructions=(
        "You are TechCorp's customer support triage agent. "
        "Determine the customer's intent and hand off to the appropriate specialist. "
        "Do NOT try to answer the question yourself."
    ),
    handoffs=[billing_agent, general_agent],
)
```

### What you observe when you run it
You run: `"I was charged twice for my CloudSync Pro subscription last month"`

You look at the AgentCI trace. This is the FIRST time you see the OpenAI SDK's trace structure through AgentCI's lens:

```
Trace
├── Span: AgentSpan (Triage Agent)
│   ├── agent: "Triage Agent"
│   ├── handoffs: ["Billing Agent", "General Agent"]
│   └── tools: []
├── Span: GenerationSpan (LLM call #1)
│   ├── model: gpt-4o-mini
│   ├── input: [system prompt + user message]
│   ├── output: [tool call: transfer_to_billing_agent]
│   └── usage: {prompt: 120, completion: 15}
├── Span: HandoffSpan                         ← THIS IS NEW
│   ├── from_agent: "Triage Agent"
│   └── to_agent: "Billing Agent"
├── Span: AgentSpan (Billing Agent)
│   ├── agent: "Billing Agent"
│   └── tools: []
├── Span: GenerationSpan (LLM call #2)
│   ├── model: gpt-4o-mini
│   ├── output: "I understand you were charged twice..."
│   └── usage: {prompt: 95, completion: 85}
└── Trace Summary
    ├── total_cost: $0.0004
    ├── handoffs: [{"from": "Triage Agent", "to": "Billing Agent"}]
    └── agents_involved: ["Triage Agent", "Billing Agent"]
```

**The big discovery:** The OpenAI SDK gives you `HandoffSpanData` with `from_agent` and `to_agent` for free. This is structured routing data that you can assert on directly. In the RAG demo, you had to infer the execution path from span ordering. Here, the SDK tells you explicitly who handed off to whom.

### What tests emerge naturally

**Test 1: "Did the triage route to the right agent?"**
```python
def test_billing_question_routes_to_billing():
    trace = run_agent("I was charged twice for my subscription")
    handoffs = trace.get_handoffs()
    assert len(handoffs) == 1
    assert handoffs[0].to_agent == "Billing Agent"
```

**Test 2: "Did the triage route a general question correctly?"**
```python
def test_general_question_routes_to_general():
    trace = run_agent("What features does CloudSync Pro include?")
    handoffs = trace.get_handoffs()
    assert handoffs[0].to_agent == "General Agent"
```

**Test 3: "Cost stays minimal for a simple route"**
```python
def test_routing_cost():
    trace = run_agent("I was charged twice")
    assert trace.total_cost < 0.005
```

### The intentional break
Change the Triage Agent's instructions from "hand off to the appropriate specialist" to "help the customer with their question directly." Now the triage agent answers itself instead of routing. Test 1 fails: `assert len(handoffs) == 1` → handoffs is empty.

**This is the exact same pattern as the RAG "silent skip" — one prompt change silently disables the core routing behavior.** Same failure mode, different framework, same AgentCI assertion catches it.

### What you discover about AgentCI
- Does the `AgentCITraceProcessor` correctly capture `HandoffSpanData`?
- Does `trace.get_handoffs()` work? (This is a NEW method you probably need to add to AgentCI)
- How does AgentCI's trace model map to the OpenAI SDK's span types? Do you need adapters?
- Does the mock system work for the OpenAI API (not just Anthropic)?

**This is the cross-framework validation moment.** If your existing `Trace`, `Span`, `ToolCall` models can accommodate the OpenAI SDK's data without schema changes, you've proven framework-agnosticism. If they can't, you'll discover exactly what abstraction is missing.

### Commit message
`feat(examples): step 1 — minimal triage with 2 agents, routing and cost assertions`

---

## Step 2: Golden Dataset of Customer Queries (1 hour)

### What you build
No agent changes. Create a structured test dataset covering all routing scenarios:

```python
GOLDEN_QUERIES = [
    # === Clear routing (should be unambiguous) ===
    {
        "query": "I was charged twice for my CloudSync Pro subscription",
        "category": "clear_billing",
        "expected_agent": "Billing Agent",
    },
    {
        "query": "What features are included in the Business plan?",
        "category": "clear_general",
        "expected_agent": "General Agent",
    },
    {
        "query": "Can I get a refund for last month?",
        "category": "clear_billing",
        "expected_agent": "Billing Agent",
    },
    {
        "query": "How do I integrate CloudSync with Slack?",
        "category": "clear_general",  # will become "technical" in Step 3
        "expected_agent": "General Agent",
    },
    
    # === Ambiguous routing (the hard cases) ===
    {
        "query": "I can't log in and I'm being charged",
        "category": "ambiguous",
        "acceptable_agents": ["Billing Agent", "General Agent"],
        # Will become more precise when we add Account + Technical agents
    },
    {
        "query": "Cancel my account and refund me",
        "category": "ambiguous",
        "acceptable_agents": ["Billing Agent", "General Agent"],
        # Two intents in one message
    },
    
    # === Edge cases ===
    {
        "query": "Hello",
        "category": "greeting",
        "expected_agent": "General Agent",
    },
    {
        "query": "Write me a poem about clouds",
        "category": "off_topic",
        # Should still route somewhere (General) or decline politely
        "expected_agent": "General Agent",
    },
    {
        "query": "I want to upgrade from Pro to Business",
        "category": "clear_billing",
        "expected_agent": "Billing Agent",
    },
    {
        "query": "Is there an API for CloudSync?",
        "category": "clear_general",  # will become "technical" in Step 3
        "expected_agent": "General Agent",
    },
]
```

### What tests emerge naturally

**Test 4: Parametrized routing correctness**
```python
@pytest.mark.parametrize("case", GOLDEN_QUERIES, ids=lambda c: c["query"][:40])
def test_routing_correctness(case):
    trace = run_agent(case["query"])
    handoffs = trace.get_handoffs()
    
    if "expected_agent" in case:
        assert handoffs[-1].to_agent == case["expected_agent"], \
            f"Expected {case['expected_agent']}, got {handoffs[-1].to_agent}"
    elif "acceptable_agents" in case:
        assert handoffs[-1].to_agent in case["acceptable_agents"], \
            f"Got {handoffs[-1].to_agent}, expected one of {case['acceptable_agents']}"
```

### What you learn
- Which queries does the 2-agent triage get wrong?
- Are the ambiguous cases actually ambiguous, or does the LLM handle them fine?
- Does the parametrized test output from AgentCI clearly show which queries failed and why?

### Commit message
`feat(examples): step 2 — golden dataset with 10 routing test cases`

---

## Step 3: Add Technical and Account Agents (1-2 hours)

### What you build
Expand from 2 to 4 specialist agents. This is where routing gets genuinely hard.

```python
technical_agent = Agent(
    name="Technical Agent",
    instructions="You handle technical issues: bugs, outages, API questions, integration help.",
    handoff_description="Customer has a technical issue, bug report, API question, or integration problem",
)

account_agent = Agent(
    name="Account Agent",
    instructions="You handle account management: password resets, 2FA, profile changes, cancellation.",
    handoff_description="Customer needs password reset, 2FA help, profile change, or account cancellation",
)

triage_agent = Agent(
    name="Triage Agent",
    instructions=(
        "You are TechCorp's customer support triage agent. "
        "Classify the customer's intent and hand off to the right specialist. "
        "Billing Agent: charges, invoices, refunds, plan upgrades/downgrades. "
        "Technical Agent: bugs, errors, API, integrations, outages. "
        "Account Agent: login issues, password reset, 2FA, cancellation. "
        "General Agent: product info, feature requests, FAQ, anything else. "
        "Do NOT answer questions yourself. Always hand off."
    ),
    handoffs=[billing_agent, technical_agent, account_agent, general_agent],
)
```

### What you observe when you run it
Now the golden dataset queries start routing differently. "How do I integrate CloudSync with Slack?" was going to General before — now it should go to Technical. "I can't log in and I'm being charged" might go to Account (login issue) or Billing (charge issue) or even Technical (bug).

### What tests emerge naturally

**Test 5: Updated routing expectations**
Update the golden dataset:
```python
{
    "query": "How do I integrate CloudSync with Slack?",
    "category": "clear_technical",
    "expected_agent": "Technical Agent",  # Changed from General
},
{
    "query": "I need to reset my password",
    "category": "clear_account",
    "expected_agent": "Account Agent",
},
```

**Test 6: "Only one handoff per query"**
```python
def test_single_handoff():
    """Triage should route once, not bounce between agents."""
    for case in GOLDEN_QUERIES:
        trace = run_agent(case["query"])
        handoffs = trace.get_handoffs()
        assert len(handoffs) == 1, \
            f"Expected 1 handoff, got {len(handoffs)} for: {case['query']}"
```

**Test 7: "The ambiguous cases resolve to specific agents"**
```python
def test_login_and_billing_routes_to_account():
    """'I can't log in and I'm being charged' — login is the primary issue."""
    trace = run_agent("I can't log in and I'm being charged")
    handoffs = trace.get_handoffs()
    # The primary intent is access — charges are secondary
    assert handoffs[0].to_agent in ["Account Agent", "Billing Agent"]
```

### The intentional break
Remove the `handoff_description` from the Account Agent. Now the LLM has no hint about what Account handles. "I need to reset my password" starts routing to General or Technical. Test 5 fails.

This demonstrates that `handoff_description` is a critical piece of routing logic — and AgentCI caught the regression when it was removed.

### What you discover about AgentCI
- With 4 possible routing targets, does AgentCI's diff engine clearly show which queries changed routing?
- When routing changes from "Account Agent" to "Technical Agent" across a model swap, is that shown clearly in the regression report?

### Commit message
`feat(examples): step 3 — expanded to 4 specialist agents with routing precision tests`

---

## Step 4: Add Tools to Specialist Agents (1-2 hours)

### What you build
Now the specialists aren't just talking — they're using tools. This is where the trace gets deeper.

```python
from agents import function_tool

@function_tool
def lookup_invoice(customer_id: str, month: str) -> str:
    """Look up a customer's invoice for a specific month."""
    # Mock implementation
    return json.dumps({"invoice_id": "INV-2024-001", "amount": 49.00, "status": "paid"})

@function_tool
def process_refund(invoice_id: str, reason: str) -> str:
    """Process a refund for a specific invoice."""
    return json.dumps({"refund_id": "REF-001", "status": "processed", "amount": 49.00})

billing_agent = Agent(
    name="Billing Agent",
    instructions="You handle billing questions. Use lookup_invoice to find charges, process_refund for refunds.",
    tools=[lookup_invoice, process_refund, change_plan],
    handoff_description="Customer has a billing, invoice, charge, or refund question",
)
```

### What you observe when you run it
The trace for "I was charged twice, can I get a refund?" now looks like:

```
Trace
├── Span: AgentSpan (Triage Agent)
├── Span: GenerationSpan (LLM #1) — decides to handoff to Billing
├── Span: HandoffSpan — Triage → Billing
├── Span: AgentSpan (Billing Agent)
├── Span: GenerationSpan (LLM #2) — decides to call lookup_invoice
├── Span: FunctionSpan (lookup_invoice)          ← NEW
│   ├── name: "lookup_invoice"
│   ├── input: {"customer_id": "C-123", "month": "2024-01"}
│   └── output: {"invoice_id": "INV-2024-001", ...}
├── Span: GenerationSpan (LLM #3) — decides to call process_refund
├── Span: FunctionSpan (process_refund)           ← NEW
│   ├── name: "process_refund"
│   ├── input: {"invoice_id": "INV-2024-001", "reason": "double charge"}
│   └── output: {"refund_id": "REF-001", "status": "processed"}
├── Span: GenerationSpan (LLM #4) — final response to customer
└── Trace Summary
    ├── total_cost: $0.003
    ├── handoffs: [{"from": "Triage", "to": "Billing"}]
    ├── tools_called: ["lookup_invoice", "process_refund"]
    └── agents_involved: ["Triage Agent", "Billing Agent"]
```

### What tests emerge naturally

**Test 8: "Billing agent uses the right tools in the right order"**
```python
def test_refund_flow_tool_sequence():
    trace = run_agent("I was charged twice, can I get a refund?")
    tools = trace.tools_called
    assert "lookup_invoice" in tools
    assert "process_refund" in tools
    # Must look up before refunding
    assert tools.index("lookup_invoice") < tools.index("process_refund")
```

**Test 9: "Technical agent creates a ticket, doesn't try to refund"**
```python
def test_technical_agent_uses_correct_tools():
    trace = run_agent("I'm getting a 500 error when I call the API")
    assert "lookup_error_code" in trace.tools_called or "create_ticket" in trace.tools_called
    assert "process_refund" not in trace.tools_called  # wrong tool for this agent
```

**Test 10: "Tool inputs are valid"**
```python
def test_refund_tool_inputs():
    trace = run_agent("Refund my January invoice please")
    refund_call = trace.get_tool_call("process_refund")
    assert refund_call is not None
    assert "invoice_id" in refund_call.input
    assert refund_call.input["invoice_id"].startswith("INV-")  # not hallucinated
```

**Test 11: "Cost scales with tool calls"**
```python
def test_complex_query_cost():
    # A refund flow has 4 LLM calls + 2 tool calls — more expensive
    trace = run_agent("I was charged twice, can I get a refund?")
    assert trace.total_cost < 0.01
    assert trace.total_cost > 0.001  # shouldn't be suspiciously cheap
```

### The AgentCI discovery moment
This is where you find out if AgentCI's `trace.tools_called` works identically for OpenAI SDK `FunctionSpan` as it does for LangGraph tool calls and raw Anthropic tool use. If the abstraction layer is right, the same assertion syntax works across all three frameworks. If it's not, you'll know exactly what mapping is missing.

### Commit message
`feat(examples): step 4 — specialist agents with tools, sequence and input assertions`

---

## Step 5: Add Guardrails (1-2 hours)

### What you build
The OpenAI Agents SDK has built-in guardrail support. Add two:

1. **Relevance guardrail** — blocks off-topic queries ("Write me a poem")
2. **PII guardrail** — flags messages containing credit card numbers or SSNs

```python
from agents import InputGuardrail, GuardrailFunctionOutput, Runner

async def relevance_guardrail(ctx, agent, input_data):
    """Check if the input is related to TechCorp customer support."""
    result = await Runner.run(guardrail_agent, input_data, context=ctx.context)
    output = result.final_output_as(RelevanceCheck)
    return GuardrailFunctionOutput(
        output_info=output,
        tripwire_triggered=not output.is_relevant,
    )

triage_agent = Agent(
    name="Triage Agent",
    instructions="...",
    handoffs=[billing_agent, technical_agent, account_agent, general_agent],
    input_guardrails=[
        InputGuardrail(guardrail_function=relevance_guardrail),
    ],
)
```

### What you observe when you run it
The trace for "Write me a poem about clouds" now shows:

```
Trace
├── Span: GuardrailSpan (relevance_guardrail)    ← NEW
│   ├── name: "relevance_guardrail"
│   └── triggered: true
└── Trace Summary
    ├── guardrails_triggered: ["relevance_guardrail"]
    ├── handoffs: []   ← no routing happened
    └── total_cost: $0.001
```

### What tests emerge naturally

**Test 12: "Off-topic queries are blocked by guardrail"**
```python
def test_off_topic_blocked():
    trace = run_agent("Write me a poem about clouds")
    assert "relevance_guardrail" in trace.guardrails_triggered
    assert len(trace.get_handoffs()) == 0  # no routing happened
```

**Test 13: "Legitimate queries pass guardrails"**
```python
def test_legitimate_query_passes_guardrails():
    trace = run_agent("I was charged twice for my subscription")
    assert len(trace.guardrails_triggered) == 0
    assert len(trace.get_handoffs()) == 1
```

**Test 14: "PII is flagged"**
```python
def test_pii_detected():
    trace = run_agent("My credit card number is 4111-1111-1111-1111, why was I charged?")
    assert "pii_guardrail" in trace.guardrails_triggered
```

### What you discover about AgentCI
- Does the trace model support `GuardrailSpanData`? You probably need `trace.guardrails_triggered` as a new property.
- How does the diff engine handle traces where a guardrail fired vs. traces where it didn't? Completely different trace shapes.
- This is the first time AgentCI captures **security-relevant behavior**. If you later pursue the agent security angle with the VC contact, these guardrail assertions are the foundation.

### Commit message
`feat(examples): step 5 — input guardrails with relevance and PII detection`

---

## Step 6: Regression Baselines + The Big Demo (1-2 hours)

### What you build
Save golden baselines for all 10+ queries. Then run the three break scenarios.

### Break Scenario A: Model Swap
Switch from `gpt-4o-mini` to `gpt-4o`. Run all golden queries.

Expected AgentCI output:
```
============ AgentCI Regression Report ============
Comparing: gpt-4o vs. baseline "gpt4o-mini"

Query: "I can't log in and I'm being charged"
  ⚠️  ROUTING_CHANGED:
    baseline: → Account Agent
    current:  → Billing Agent
    Note: gpt-4o weighted "being charged" more heavily than "can't log in"

Query: "Cancel my account and refund me"
  ⚠️  ROUTING_CHANGED:
    baseline: → Account Agent
    current:  → Billing Agent
    Note: Different intent prioritization

Query: "Can I get a refund for last month?"
  ✅ ROUTING: identical → Billing Agent
  ⚠️  COST_SPIKE: $0.0004 → $0.0025 (6.2x)
  ✅ TOOLS: identical [lookup_invoice, process_refund]

SUMMARY: 2 routing changes, 3 cost spikes across 10 queries
```

### Break Scenario B: Prompt Change
Remove "Do NOT answer questions yourself. Always hand off." from the triage prompt.

Expected result: Triage starts answering directly. Handoff count drops from 1 to 0 for some queries. Tests 1-7 start failing.

### Break Scenario C: Specialist Confusion
Swap the `handoff_description` between Billing and Technical agents. Now billing queries route to Technical and vice versa.

Expected result: 4-5 routing assertions fail. The diff report shows a clear pattern: Billing↔Technical swap.

### Commit message
`feat(examples): step 6 — regression baselines with model swap, prompt break, and routing swap demos`

---

## Step 7: Mock System + Polish (2-3 hours)

### What you build
Record all golden traces into mock fixtures so everything runs with zero API keys. Package with README, Makefile, GitHub Actions.

### The mock challenge for OpenAI SDK
Unlike the Anthropic mocking in DevAgent, here you need to mock the OpenAI Responses API or Chat Completions API. The OpenAI Agents SDK calls `openai.chat.completions.create()` under the hood. Your mock needs to intercept at that layer.

**Option A:** Use the SDK's built-in model override — the Agents SDK supports custom model providers. You could create an `AgentCIMockModel` that returns canned responses.

**Option B:** Mock at the HTTP layer with `httpx` mocking (same pattern as DevAgent).

Either way, the mock system needs to handle the handoff tool calls specifically — when the LLM returns `transfer_to_billing_agent` as a tool call, the mock needs to return that exact function call structure.

### What you discover about AgentCI
- Does the mock system generalize from Anthropic to OpenAI? If you built it specifically for Anthropic's API format, it needs adapters now.
- Can `agentci init` detect that this project uses OpenAI (not Anthropic) and generate the right mock setup?

### Deliverables
- `make test` runs all tests in <3 seconds with zero API keys
- `make test-live` runs against real OpenAI API
- `make demo-break` shows the model swap regression
- README with 60-second quickstart
- GitHub Actions workflow

### Commit message
`feat(examples): step 7 — mock system, README, CI workflow for Support Router`

---

## Summary: What You Build vs. What You Discover

| Step | Agent Capability | Test Discovered | AgentCI Gap Found |
|------|-----------------|-----------------|-------------------|
| 0 | Scaffold | — | — |
| 1 | 2-agent triage | Routing assertion, cost guard | `AgentCITraceProcessor` for OpenAI SDK, `trace.get_handoffs()` |
| 2 | Golden dataset | Parametrized routing tests | Routing diff output format |
| 3 | 4-agent triage | Multi-target routing, single-handoff check | Diff engine for routing changes |
| 4 | Tools on specialists | Tool sequence within specialist, input validation | `trace.tools_called` cross-framework parity |
| 5 | Guardrails | Guardrail triggered/not-triggered assertions | `trace.guardrails_triggered`, security span capture |
| 6 | Baselines + breaks | Regression detection across model/prompt/config | Routing-aware diff report |
| 7 | Mocks + polish | Mock parity | OpenAI mock adapters |

## Time Estimate

| Step | Time | Running Total |
|------|------|---------------|
| Step 0: Scaffold | 30 min | 30 min |
| Step 1: 2-agent triage | 1-2 hrs | 2.5 hrs |
| Step 2: Golden dataset | 1 hr | 3.5 hrs |
| Step 3: 4-agent triage | 1-2 hrs | 5 hrs |
| Step 4: Tools on specialists | 1-2 hrs | 7 hrs |
| Step 5: Guardrails | 1-2 hrs | 9 hrs |
| Step 6: Baselines + breaks | 1-2 hrs | 11 hrs |
| Step 7: Mocks + polish | 2-3 hrs | 14 hrs |

**Total: ~2-3 days of focused work**

---

## What Makes This Demo Different From RAG and DevAgent

| Dimension | RAG Demo | DevAgent Demo | Support Router Demo |
|-----------|----------|---------------|---------------------|
| Framework | LangGraph | Raw Python + Anthropic | OpenAI Agents SDK |
| Pattern | Retrieve → Generate | Multi-tool sequential | Classify → Route → Handoff |
| Key assertion | `assert "vector_search" in trace.tools_called` | `assert trace.tool_sequence matches expected` | `assert handoffs[0].to_agent == "Billing"` |
| Unique AgentCI feature tested | Execution path matching | Chaos testing / error injection | Handoff tracing + guardrail capture |
| Killer break scenario | Prompt change skips retrieval | Prompt injection causes semantic drift | Model swap changes routing decisions |

After this demo, your portfolio covers:
- **3 frameworks** (LangGraph, raw Anthropic, OpenAI Agents SDK)
- **3 patterns** (retrieve+generate, multi-tool sequential, classify+route+handoff)
- **75%+ of the agent market** by pattern coverage
- **Cross-framework proof** that the same assertion primitives work everywhere

That's the launch portfolio.