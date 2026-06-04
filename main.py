"""
A multi-agent LangGraph example for learning.

Agents collaborate with PARALLEL research, an automated REVIEW LOOP, a
HUMAN-IN-THE-LOOP approval gate, STREAMING output, and DURABLE memory
(SQLite checkpointing):

    START ─┬─> research_environment ─┐
           ├─> research_economy      ├─> combine ─> writer ─> reviewer ─┐
           └─> research_science  ────┘                ^                 │
                                                       └── needs work ──┘
                                                                │ (approved / capped)
                                                                v
                                          human_gate ──approve──> finalize ─> END
                                              ^   │
                                              └───┘ (human sends feedback -> writer)

  - Researchers : three agents each research a different ANGLE, concurrently.
  - Combine     : merges their notes into one research brief (fan-in).
  - Writer      : writes the article (and revises using feedback).
  - Reviewer    : AI editor. Approve/cap -> human_gate. Otherwise -> back to writer.
  - Human gate  : PAUSES for a human to approve or hand back feedback.
  - Finalize    : publishes the approved draft.

Powered by z.ai (GLM) through the OpenAI-compatible API.

--- LangGraph ideas demonstrated ---

* State + reducers: nodes RETURN partial updates (never mutate). `research_notes`
  is written by three parallel nodes, so it uses an `operator.add` reducer to
  merge their writes; single-writer fields use the default "overwrite".

* Messages: each agent prompts with a SystemMessage (who it IS) + HumanMessage.

* Conditional edges + cycles: routing functions loop writer<->reviewer (capped by
  MAX_REVISIONS) and writer<->human_gate, so the graph always terminates.

* Structured output: the reviewer returns a typed Pydantic object so routing is
  reliable (function-calling method, which z.ai/GLM supports).

* Streaming: app.stream(..., stream_mode="updates") shows each node's output live.

* Human-in-the-loop: the human_gate node calls interrupt() to PAUSE the graph;
  the caller inspects the draft and resumes with Command(resume=<decision>).

* Durable memory: compiling with SqliteSaver persists every checkpoint to a file
  on disk, so a thread survives process restarts and can be inspected/resumed.

* Visualization: app.get_graph().draw_mermaid() renders the graph as a diagram.
"""

import operator
import os
import sqlite3
import uuid
from typing import Annotated, TypedDict

from pydantic import BaseModel, Field

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, END, StateGraph
from langgraph.types import interrupt, Command


# ---------------------------------------------------------------------------
# 1. The LLM
# ---------------------------------------------------------------------------
# z.ai exposes an OpenAI-compatible endpoint, so we use ChatOpenAI and just
# point it at z.ai's base URL. The API key is read from the environment
# (export ZAI_API_KEY="..." in your shell) -- never hard-code secrets.
llm = ChatOpenAI(
    model="glm-5.1",
    api_key=os.environ["ZAI_API_KEY"],
    base_url="https://api.z.ai/api/coding/paas/v4",  # coding-plan endpoint (vs pay-as-you-go: .../api/paas/v4)
    temperature=0.7,
)

# Caps the automated writer<->reviewer loop so it always terminates.
MAX_REVISIONS = 3

# Where durable checkpoints are stored. Delete this file to wipe all memory.
CHECKPOINT_DB = "checkpoints.sqlite"

# The angles our three parallel researchers each focus on.
RESEARCH_ANGLES = {
    "environment": "the environmental and ecological importance",
    "economy": "the economic importance",
    "science": "the scientific and biological facts",
}


# ---------------------------------------------------------------------------
# 2. Shared state
# ---------------------------------------------------------------------------
# `research_notes` is written by THREE researchers at once, so it needs a
# reducer: `operator.add` concatenates the lists instead of letting one
# parallel write overwrite another. Every other field is written by a single
# node, so it uses the default "overwrite" behaviour.
class State(TypedDict):
    topic: str                                          # input topic
    research_notes: Annotated[list[str], operator.add]  # appended by each researcher (parallel)
    research: str                                       # merged brief (combine step)
    draft: str                                          # written / revised by the writer
    feedback: str                                       # critique fed back to the writer
    approved: bool                                      # latest verdict (AI or human)
    revisions: int                                      # how many drafts the writer has produced
    final: str                                          # the published result


# ---------------------------------------------------------------------------
# 3. Structured output for the reviewer's decision
# ---------------------------------------------------------------------------
# Forcing the model to fill this schema gives us a reliable bool to route on,
# instead of parsing words like "approved" out of free-form text.
class Review(BaseModel):
    approved: bool = Field(
        description="True only if the draft is clear, accurate, and needs no further edits."
    )
    feedback: str = Field(
        description="Specific, actionable suggestions for the writer. Empty string if approved."
    )


# method="function_calling" routes through GLM's tool-calling support, which
# z.ai handles reliably (its default json_schema mode is not fully supported).
reviewer_llm = llm.with_structured_output(Review, method="function_calling")


# ---------------------------------------------------------------------------
# 4. System prompts (each agent's standing identity / instructions)
# ---------------------------------------------------------------------------
RESEARCHER_SYSTEM = (
    "You are a meticulous researcher. You distill a topic into a few sharp, "
    "factual bullet points. Be concise and avoid fluff."
)
WRITER_SYSTEM = (
    "You are a skilled writer. You turn research notes into clear, engaging "
    "prose for a general audience, and you act on editor feedback when given."
)
REVIEWER_SYSTEM = (
    "You are a demanding editor. Judge whether the draft is publication-ready: "
    "clear, accurate, well-structured, and roughly 120 words. Approve only if it "
    "truly needs no changes; otherwise give specific, actionable feedback."
)


# ---------------------------------------------------------------------------
# 5. The agents (each is a function: State -> partial State)
# ---------------------------------------------------------------------------
def make_researcher(name: str, desc: str):
    """Factory: build one researcher node focused on a given angle.

    All three returned nodes write to `research_notes`; the reducer on that
    field merges their parallel outputs.
    """
    def _researcher(state: State) -> dict:
        print(f"🔎 Researcher [{name}] working...")
        messages = [
            SystemMessage(content=RESEARCHER_SYSTEM),
            HumanMessage(content=(
                f"Give 2-3 concise bullet points about {desc} of this topic.\n\n"
                f"Topic: {state['topic']}"
            )),
        ]
        response = llm.invoke(messages)
        # Return a one-item list; the reducer appends it to the others.
        return {"research_notes": [f"### {name.title()}\n{response.content}"]}

    return _researcher


def combine_research(state: State) -> dict:
    """Fan-in: merge the parallel researchers' notes into one brief."""
    print("🧩 Combining research from all angles...")
    brief = "\n\n".join(state["research_notes"])
    return {"research": brief}


def writer(state: State) -> dict:
    """Write the first draft, or revise using feedback (from AI or human)."""
    revisions = state.get("revisions", 0)
    feedback = state.get("feedback", "")

    if feedback:
        print(f"✍️  Writer revising (revision {revisions + 1})...")
        task = (
            f"Revise your draft about '{state['topic']}' using the feedback "
            f"below. Return only the improved paragraph.\n\n"
            f"Feedback:\n{feedback}\n\n"
            f"Current draft:\n{state['draft']}\n\n"
            f"Research brief:\n{state['research']}"
        )
    else:
        print("✍️  Writer drafting...")
        task = (
            f"Write a single clear, engaging paragraph (about 120 words) about "
            f"'{state['topic']}', using this research brief:\n\n{state['research']}"
        )

    response = llm.invoke([SystemMessage(content=WRITER_SYSTEM), HumanMessage(content=task)])
    return {"draft": response.content, "revisions": revisions + 1}


def reviewer(state: State) -> dict:
    """AI editor: judge the draft and return a typed verdict + feedback."""
    print("🧐 Reviewer evaluating...")
    review: Review = reviewer_llm.invoke([
        SystemMessage(content=REVIEWER_SYSTEM),
        HumanMessage(content=f"Review this draft:\n\n{state['draft']}"),
    ])
    verdict = "✅ approved" if review.approved else "🔁 needs revision"
    print(f"    -> {verdict}")
    return {"approved": review.approved, "feedback": review.feedback}


def human_gate(state: State) -> dict:
    """Human-in-the-loop: PAUSE and let a person approve or send feedback.

    interrupt() stops the graph and surfaces this payload to the caller. The
    graph resumes when the caller sends Command(resume=<decision>); that value
    becomes interrupt()'s return. NOTE: a node that interrupts re-runs from the
    top on resume, so keep pre-interrupt logic minimal/idempotent.
    """
    decision = interrupt({
        "question": "Approve for publication, or type feedback to send it back to the writer.",
        "ai_approved": state["approved"],
        "draft": state["draft"],
    })

    text = (decision or "").strip()
    if text.lower() in ("", "approve", "approved", "yes", "y", "ok", "publish"):
        print("👍 Human approved.")
        return {"approved": True, "feedback": ""}

    print("📝 Human requested changes.")
    return {"approved": False, "feedback": text}


def finalize(state: State) -> dict:
    """Publish the approved draft as the final result."""
    print("🏁 Finalizing approved draft.")
    return {"final": state["draft"]}


# ---------------------------------------------------------------------------
# 6. Routing logic (conditional edges) -- pure functions, no LLM calls
# ---------------------------------------------------------------------------
def route_after_review(state: State) -> str:
    """After the AI reviewer: keep revising, or hand off to the human."""
    if state["approved"] or state["revisions"] >= MAX_REVISIONS:
        return "human_gate"
    return "writer"  # loop back for an automated revision


def route_after_human(state: State) -> str:
    """After the human gate: publish, or send the draft back for a rewrite."""
    return "finalize" if state["approved"] else "writer"


# ---------------------------------------------------------------------------
# 7. Build the graph (wire the agents together)
# ---------------------------------------------------------------------------
graph = StateGraph(State)

# One researcher node per angle (created by the factory).
for angle_name, angle_desc in RESEARCH_ANGLES.items():
    graph.add_node(f"research_{angle_name}", make_researcher(angle_name, angle_desc))

graph.add_node("combine", combine_research)
graph.add_node("writer", writer)
graph.add_node("reviewer", reviewer)
graph.add_node("human_gate", human_gate)
graph.add_node("finalize", finalize)

# Fan-out: START -> all researchers (parallel). Fan-in: each -> combine.
for angle_name in RESEARCH_ANGLES:
    graph.add_edge(START, f"research_{angle_name}")
    graph.add_edge(f"research_{angle_name}", "combine")

graph.add_edge("combine", "writer")
graph.add_edge("writer", "reviewer")

graph.add_conditional_edges(
    "reviewer", route_after_review, {"writer": "writer", "human_gate": "human_gate"}
)
graph.add_conditional_edges(
    "human_gate", route_after_human, {"writer": "writer", "finalize": "finalize"}
)
graph.add_edge("finalize", END)


# ---------------------------------------------------------------------------
# 8. Helpers for the demo
# ---------------------------------------------------------------------------
def stream_until_pause(app, payload, config) -> dict | None:
    """Stream a run (or a resume), printing node updates as they arrive.

    Returns the interrupt payload if the graph PAUSED for human input,
    otherwise None (the run reached END).
    """
    for event in app.stream(payload, config=config, stream_mode="updates"):
        if "__interrupt__" in event:
            return event["__interrupt__"][0].value
        for node_name, delta in event.items():
            print(f"  ↳ [{node_name}] updated: {', '.join(delta.keys())}")
    return None


def ask_human(payload: dict) -> str:
    """Show the interrupt payload and collect the human's decision.

    Falls back to auto-approve when there's no interactive terminal (e.g. piped
    input / CI), so the script still completes end-to-end.
    """
    print("\n" + "🟡 HUMAN INPUT NEEDED ".ljust(60, "-"))
    print("AI already approved:", payload["ai_approved"])
    print("\nDraft:\n", payload["draft"])
    print("\n" + payload["question"])
    try:
        answer = input("> ").strip()
    except EOFError:
        print("[no terminal -> auto-approving]")
        return "approve"
    return answer or "approve"


# ---------------------------------------------------------------------------
# 9. Run it -- SQLite memory + streaming + human-in-the-loop + diagram
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Print the graph as a Mermaid diagram (paste into https://mermaid.live or a
    # Markdown file). To render a PNG instead: app.get_graph().draw_mermaid_png().
    # (Compiling once without a checkpointer just to draw avoids needing config.)
    print("=== Graph (Mermaid) ===")
    print(graph.compile().get_graph().draw_mermaid())

    # check_same_thread=False because LangGraph runs parallel nodes on threads.
    conn = sqlite3.connect(CHECKPOINT_DB, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    app = graph.compile(checkpointer=checkpointer)

    # A fresh thread per run so reruns start clean. Hard-code a fixed id instead
    # to resume an earlier run from the SQLite file across restarts.
    thread_id = f"bees-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}
    print(f"\n=== Streaming run (thread: {thread_id}) ===\n")

    pause = stream_until_pause(
        app, {"topic": "Why bees are important for the planet"}, config
    )
    # Resume loop: keep going while the graph pauses at the human gate.
    while pause is not None:
        human_decision = ask_human(pause)
        pause = stream_until_pause(app, Command(resume=human_decision), config)

    final_state = app.get_state(config).values
    print("\n" + "=" * 60)
    print(f"REVISIONS MADE: {final_state['revisions']}  |  APPROVED: {final_state['approved']}")
    print("\nFINAL:\n", final_state["final"])

    history = list(app.get_state_history(config))
    print("\n" + "-" * 60)
    print(f"💾 Durable checkpoints in {CHECKPOINT_DB} for thread '{thread_id}': {len(history)}")
    print("   This file persists across restarts -- reuse the thread_id to resume.")

    conn.close()
