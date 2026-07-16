"""LangGraph StateGraph for the returns-concierge interview.

Replaces the hand-rolled state machine in the old ``ConciergeSession`` with an
explicit graph of three nodes::

    agent  →  (route)  →  ask_customer  →  agent  (loop)
                       →  finalize      →  END

The ``ConciergeSession`` class at the bottom wraps the graph with the same
``start()`` / ``answer()`` API so every upstream script works unchanged.
"""

from __future__ import annotations

import json
import uuid
from typing import Annotated, Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt

from src.concierge.concierge import (
    MAX_QUESTIONS,
    build_system_prompt,
    classify_case,
    enrich_diagnosis,
    resolve_materials,
)
from src.concierge.portkey_llm import make_llm, provider_name
from src.concierge.tools import ask_question, submit_diagnosis
from src.physics.fabric_ontology import FabricOntology

# ------------------------------------------------------------------ state

class ConciergeState(dict):
    """Typed state for the concierge graph.

    Using a plain dict subclass so LangGraph can serialise it.  The
    ``messages`` key uses the ``add_messages`` reducer (appends new messages
    instead of replacing the list).
    """

ConciergeState.__annotations__ = {
    "messages": Annotated[list[BaseMessage], add_messages],
    "questions_asked": int,
    "transcript": list[dict],
    "result": dict,
}

# ------------------------------------------------------------------ graph

_BOTH_TOOLS = [ask_question, submit_diagnosis]
_DIAGNOSIS_ONLY = [submit_diagnosis]


def _build_graph(
    llm: Any,
    ontology: FabricOntology,
    claimed_materials: list[str],
    weaves: list[str],
) -> StateGraph:
    """Construct (but do NOT compile) the concierge interview graph."""

    # ---- node: agent ----
    def agent(state: ConciergeState) -> dict:
        if state["questions_asked"] >= MAX_QUESTIONS:
            model = llm.bind_tools(_DIAGNOSIS_ONLY, tool_choice="required")
            # Inject a nudge so the model understands why it can only diagnose.
            extra = [HumanMessage(
                content=("IMPORTANT: You have used all your questions. "
                         "Submit the diagnosis now."))]
        else:
            model = llm.bind_tools(_BOTH_TOOLS, tool_choice="required")
            extra = []
        response = model.invoke(list(state["messages"]) + extra)
        return {"messages": [response]}

    # ---- node: ask_customer ----
    def ask_customer(state: ConciergeState) -> dict:
        last_msg: AIMessage = state["messages"][-1]
        tc = last_msg.tool_calls[0]
        question_data = tc["args"]

        # Interrupt: the caller provides the customer's answer when resuming.
        answer: str = interrupt({"type": "question", **question_data})

        tool_msg = ToolMessage(
            content=json.dumps({"customer_answer": answer}),
            tool_call_id=tc["id"],
        )
        return {
            "messages": [tool_msg],
            "questions_asked": state["questions_asked"] + 1,
            "transcript": state["transcript"] + [
                {"role": "concierge", **question_data},
                {"role": "customer", "text": answer},
            ],
        }

    # ---- node: finalize ----
    def finalize(state: ConciergeState) -> dict:
        last_msg: AIMessage = state["messages"][-1]
        tc = last_msg.tool_calls[0]
        payload = tc["args"]

        enriched = enrich_diagnosis(
            payload, ontology, claimed_materials, weaves)

        tool_msg = ToolMessage(
            content=json.dumps({"status": "diagnosis_submitted"}),
            tool_call_id=tc["id"],
        )
        return {
            "messages": [tool_msg],
            "transcript": state["transcript"] + [
                {"role": "diagnosis", **enriched}],
            "result": {"type": "diagnosis", "data": enriched},
        }

    # ---- routing ----
    def route_after_agent(state: ConciergeState) -> str:
        last_msg = state["messages"][-1]
        if not hasattr(last_msg, "tool_calls") or not last_msg.tool_calls:
            # Model returned no tool call — shouldn't happen with
            # tool_choice="required", but handle gracefully.
            return "agent"  # retry
        name = last_msg.tool_calls[0]["name"]
        if name == "submit_diagnosis":
            return "finalize"
        return "ask_customer"

    # ---- wire the graph ----
    graph = StateGraph(ConciergeState)
    graph.add_node("agent", agent)
    graph.add_node("ask_customer", ask_customer)
    graph.add_node("finalize", finalize)

    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", route_after_agent, {
        "agent": "agent",
        "ask_customer": "ask_customer",
        "finalize": "finalize",
    })
    graph.add_edge("ask_customer", "agent")
    graph.add_edge("finalize", END)

    return graph


# --------------------------------------------------------------- session API

class ConciergeSession:
    """One return interview — drop-in replacement for the old class.

    Same public API::

        session = ConciergeSession(product, ontology, diagnosis_row)
        event = session.start()           # {"type": "question", ...}
        event = session.answer("text")    # {"type": "question"|"diagnosis", ...}

    Event shapes are identical to before so the dashboard and insights store
    work unchanged.
    """

    def __init__(
        self,
        product: dict,
        ontology: FabricOntology,
        diagnosis_row: dict | None = None,
        category: str | None = None,
        *,
        provider: str | None = None,
        llm: Any | None = None,
        callbacks: list | None = None,
    ):
        self._llm = llm or make_llm(provider, callbacks=callbacks)
        self.ontology = ontology
        self.claimed_materials, self.weaves, _ = resolve_materials(
            product, ontology, category)
        self._system = build_system_prompt(
            product, ontology, diagnosis_row, category)

        graph = _build_graph(
            self._llm, ontology, self.claimed_materials, self.weaves)
        self._compiled = graph.compile(checkpointer=MemorySaver())
        self._thread = {"configurable": {"thread_id": uuid.uuid4().hex}}

        self.transcript: list[dict] = []
        self.questions_asked: int = 0
        self._finished = False

    # ---- properties for backward-compat with scripts ----

    @property
    def model_id(self) -> str:
        return getattr(self._llm, "model_name",
                       getattr(self._llm, "model_id", "unknown"))

    @property
    def provider(self) -> str:
        return provider_name()

    @property
    def transport(self) -> str:
        """Replaces the old ``mode`` attribute ("tools"/"json")."""
        return "langgraph"

    # Alias for backward compat (run_llmops reads session.mode)
    @property
    def mode(self) -> str:
        return self.transport

    # ---- public interview API ----

    def start(self) -> dict:
        initial_state: dict = {
            "messages": [
                SystemMessage(content=self._system),
                HumanMessage(content=(
                    "The customer just clicked 'Return item'. "
                    "Begin the interview.")),
            ],
            "questions_asked": 0,
            "transcript": [],
            "result": {},
        }
        return self._run(initial_state)

    def answer(self, text: str) -> dict:
        if self._finished:
            raise RuntimeError("Session already finished.")
        return self._run(Command(resume=text))

    # ---- internals ----

    def _run(self, input_val: Any) -> dict:
        # Stream the graph; it will either interrupt (question) or complete
        # (diagnosis).
        for _chunk in self._compiled.stream(
                input_val, self._thread, stream_mode="values"):
            pass  # drive the graph to completion or interrupt

        state = self._compiled.get_state(self._thread)

        # Interrupted → question pending
        if state.next:
            for task in state.tasks:
                if hasattr(task, "interrupts") and task.interrupts:
                    question_event = task.interrupts[0].value
                    self.questions_asked = state.values.get(
                        "questions_asked", self.questions_asked)
                    self.transcript = state.values.get(
                        "transcript", self.transcript)
                    return question_event

        # Graph completed → diagnosis ready
        self._finished = True
        self.questions_asked = state.values.get(
            "questions_asked", self.questions_asked)
        self.transcript = state.values.get("transcript", self.transcript)
        return state.values.get("result", {})
