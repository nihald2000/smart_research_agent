# smart_research_agent Question Answering Agent with Gemini and LangGraph

This project implements an agent that attempts to answer complex, multi-hop questions (similar to the GAIA benchmark) using Google's Gemini APIs, orchestrated by the **LangGraph** framework.

## Core Functionality

*   **Stateful Multi-Step Reasoning:** Uses LangGraph to manage the state (query, plan, intermediate results) across multiple steps.
*   **Clear Control Flow:** Defines nodes for planning, execution (tool use simulation), and synthesis, with conditional edges managing the process.
*   **Information Retrieval:** Leverages Gemini's knowledge base.
*   **Image Analysis (Potential):** Can incorporate Gemini Pro Vision via a dedicated step if needed.
*   **Structured Output:** Synthesizes the final answer based on intermediate findings and original query requirements.

## Approach with LangGraph

*   **State:** A dictionary (`AgentState`) tracks the query, plan, current step, results, and errors.
*   **Nodes:** Python functions representing key actions:
    *   `plan_step`: Generates the execution plan.
    *   `execute_tool_step`: Executes the current step of the plan (simulating tool use via Gemini calls).
    *   `synthesize_result`: Combines results into the final answer.
*   **Edges:** Control logic determines the next node based on the current state (e.g., if planning failed, if more steps exist, if errors occurred).
