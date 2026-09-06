"""Candidate-added regression tests for high-value contract boundaries."""

from __future__ import annotations

import json
import pathlib
import sys
from html import escape
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

SRC = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))

from agent import PlanRun, StepResult, run_planning_agent  # noqa: E402
from planner import Step, revise_plan  # noqa: E402


def _step(n: int, goal: str, hint: str = "none") -> Step:
    return Step(n=n, goal=goal, tool_hint=hint)


def _result(step: Step, observation: str = "ok", text: str | None = None) -> StepResult:
    return StepResult(
        step=step,
        status="done",
        text=text or f"result for {step.goal}",
        observation=observation,
    )


def _called_step(call: Any) -> Step:
    if "step" in call.kwargs:
        return call.kwargs["step"]
    return call.args[0]


def test_caller_supplied_plan_is_executed_without_redrafting():
    approved_plan = [
        _step(1, "Gather facts", "web_search"),
        _step(2, "Write the answer"),
    ]
    results = [_result(approved_plan[0]), _result(approved_plan[1], text="final")]

    with patch("agent.write_plan") as draft, patch(
        "agent.execute_step", side_effect=results
    ) as execute:
        run = run_planning_agent("Use this plan", plan=approved_plan, max_revisions=0)

    draft.assert_not_called()
    assert [_called_step(call) for call in execute.call_args_list] == approved_plan
    assert [result.step for result in run.step_results] == approved_plan
    assert run.stopped_reason == "done"


def test_explicit_empty_caller_plan_is_an_error_without_redrafting():
    with patch("agent.write_plan") as draft, patch("agent.execute_step") as execute:
        run = run_planning_agent("Use the supplied plan", plan=[])

    draft.assert_not_called()
    execute.assert_not_called()
    assert run.stopped_reason == "error"
    assert "couldn't draft a plan" in run.final_answer


def test_replanning_does_not_mutate_the_caller_owned_plan():
    approved_plan = [
        _step(1, "Check conditions", "web_search"),
        _step(2, "Old research", "fetch_url"),
        _step(3, "Old final answer"),
    ]
    original_objects = tuple(approved_plan)
    original_fields = [
        (step.n, step.goal, step.tool_hint) for step in approved_plan
    ]
    revised = [
        _step(2, "New research", "web_search"),
        _step(3, "New final answer"),
    ]
    results = [
        _result(approved_plan[0], "surprise: conditions changed"),
        _result(revised[0]),
        _result(revised[1], text="new final"),
    ]

    with patch("agent.write_plan") as draft, patch(
        "agent.revise_plan", return_value=revised
    ), patch("agent.execute_step", side_effect=results):
        run = run_planning_agent("Respect my plan", plan=approved_plan)

    draft.assert_not_called()
    assert len(approved_plan) == 3
    assert tuple(approved_plan) == original_objects
    assert all(actual is original for actual, original in zip(approved_plan, original_objects))
    assert [
        (step.n, step.goal, step.tool_hint) for step in approved_plan
    ] == original_fields
    assert [result.step.goal for result in run.step_results] == [
        "Check conditions",
        "New research",
        "New final answer",
    ]


def test_zero_revision_budget_keeps_executing_the_original_future():
    plan = [
        _step(1, "Check conditions", "web_search"),
        _step(2, "Write the original answer"),
    ]
    results = [
        _result(plan[0], "surprise: unexpected weather"),
        _result(plan[1], text="original final"),
    ]

    with patch("agent.revise_plan") as revise, patch(
        "agent.execute_step", side_effect=results
    ) as execute:
        run = run_planning_agent(
            "Do not revise", plan=plan, max_revisions=0
        )

    revise.assert_not_called()
    assert [_called_step(call) for call in execute.call_args_list] == plan
    assert run.revisions == []
    assert run.final_answer == "original final"
    assert run.stopped_reason == "done"


def test_execute_step_does_not_exceed_budget_when_model_batches_tool_calls():
    """A single assistant reply may request more tools than the budget allows."""
    from agent import execute_step

    model_reply = SimpleNamespace(
        content="",
        tool_calls=[
            SimpleNamespace(
                id="first",
                function=SimpleNamespace(
                    name="web_search", arguments=json.dumps({"query": "first"})
                ),
            ),
            SimpleNamespace(
                id="second",
                function=SimpleNamespace(
                    name="web_search", arguments=json.dumps({"query": "second"})
                ),
            ),
        ],
    )
    executed_queries: list[str] = []

    def fake_search(query: str) -> str:
        executed_queries.append(query)
        return "[]"

    with patch("agent.chat_with_tools", return_value=model_reply), patch(
        "agent.chat", return_value="short result"
    ), patch("agent._observe", return_value="ok"), patch(
        "agent.TOOL_DISPATCH", {"web_search": fake_search}
    ):
        result = execute_step(
            _step(1, "Search", "web_search"), "Goal", max_tool_calls=1
        )

    assert executed_queries == ["first"]
    assert [call.name for call in result.tool_calls] == ["web_search"]


def test_revision_audit_keeps_true_before_and_after_snapshots():
    initial = [
        _step(1, "Check conditions", "web_search"),
        _step(2, "Old research", "fetch_url"),
        _step(3, "Old final answer"),
    ]
    revised = [
        _step(2, "New research", "web_search"),
        _step(3, "New final answer"),
    ]
    trigger = "surprise: the original route is closed"
    results = [
        _result(initial[0], trigger),
        _result(revised[0]),
        _result(revised[1], text="new final"),
    ]

    with patch("agent.revise_plan", return_value=revised), patch(
        "agent.execute_step", side_effect=results
    ):
        run = run_planning_agent("Keep an audit", plan=initial)

    assert run.revisions == [
        {
            "after_step": initial[0].n,
            "trigger": trigger,
            "before": [step.to_dict() for step in initial[1:]],
            "after": [step.to_dict() for step in revised],
        }
    ]


def test_reviser_receives_the_triggering_step_in_done_and_only_future_remaining():
    current = _step(10, "Inspect the route", "web_search")
    future = [
        _step(11, "Read the original source", "fetch_url"),
        _step(12, "Write the answer"),
    ]
    plan = [current, *future]
    trigger = "surprise: the route is closed"
    captured: dict[str, Any] = {}

    def spy_revise(
        goal: str,
        done: list[tuple[Step, str]],
        remaining: list[Step],
        observation: str,
        max_steps: int,
    ) -> list[Step]:
        captured.update(
            goal=goal,
            done=done,
            remaining=remaining,
            observation=observation,
            max_steps=max_steps,
        )
        return list(remaining)

    results = [_result(current, trigger), _result(future[0]), _result(future[1])]
    with patch("agent.revise_plan", side_effect=spy_revise), patch(
        "agent.execute_step", side_effect=results
    ):
        run = run_planning_agent("Respect past and future", plan=plan, max_steps=5)

    assert captured["done"] == [(current, trigger)]
    assert captured["remaining"] == future
    assert captured["observation"] == trigger
    assert run.stopped_reason == "done"


def test_revise_plan_model_exception_preserves_approved_future_without_mutation():
    done = [(_step(10, "Completed search", "web_search"), "surprise: new constraint")]
    remaining = [
        _step(11, "Keep this research", "fetch_url"),
        _step(12, "Keep this final answer"),
    ]
    original_objects = tuple(remaining)
    original_fields = [
        (step.n, step.goal, step.tool_hint) for step in remaining
    ]

    with patch("planner.chat", side_effect=RuntimeError("model unavailable")) as chat:
        revised = revise_plan(
            "Preserve approved work",
            done=done,
            remaining=remaining,
            observation="surprise: new constraint",
        )

    chat.assert_called_once()
    assert revised == list(original_objects)
    assert len(remaining) == 2
    assert all(actual is original for actual, original in zip(remaining, original_objects))
    assert [
        (step.n, step.goal, step.tool_hint) for step in remaining
    ] == original_fields


def test_revise_plan_normalizes_model_numbers_and_counts_capacity_by_done_length():
    done = [(_step(10, "Completed search", "web_search"), "surprise: changed")]
    remaining = [
        _step(11, "Old research", "fetch_url"),
        _step(12, "Old image", "generate_image"),
        _step(13, "Old answer"),
    ]
    reply = json.dumps(
        {
            "steps": [
                {"n": 99, "goal": "New research", "tool_hint": "web_search"},
                {"n": 100, "goal": "New image", "tool_hint": "generate_image"},
                {"n": 101, "goal": "New answer", "tool_hint": "none"},
            ]
        }
    )

    with patch("planner.chat", return_value=reply):
        revised = revise_plan(
            "Renumber the future",
            done=done,
            remaining=remaining,
            observation="surprise: changed",
            max_steps=4,
        )

    assert [step.n for step in revised] == [11, 12, 13]
    assert [step.goal for step in revised] == [
        "New research",
        "New image",
        "New answer",
    ]
    assert len(done) + len(revised) == 4


class _Expander:
    def __init__(self, events: list[tuple[str, str]], label: str) -> None:
        self._events = events
        self._label = label

    def __enter__(self) -> "_Expander":
        self._events.append(("expander_enter", self._label))
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self._events.append(("expander_exit", self._label))
        return False


class _RecordingStreamlit:
    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    def markdown(self, body: str, **kwargs: Any) -> None:
        self.events.append(("markdown", body))

    def text(self, body: str, **kwargs: Any) -> None:
        self.events.append(("text", body))

    def caption(self, body: str, **kwargs: Any) -> None:
        self.events.append(("caption", body))

    def expander(self, label: str, **kwargs: Any) -> _Expander:
        self.events.append(("expander", label))
        return _Expander(self.events, label)

    def image(self, image: str, **kwargs: Any) -> None:
        self.events.append(("image", image))


def test_revision_renders_after_its_real_triggering_step_and_before_the_next(monkeypatch):
    import itinerary_app

    first = _step(10, "Discover the constraint", "web_search")
    second = _step(11, "Use the revised plan")
    run = PlanRun(
        goal="Show the timeline",
        initial_plan=[first, second],
        final_plan=[first, second],
        step_results=[_result(first, "surprise: changed"), _result(second)],
        revisions=[
            {
                "after_step": 10,
                "trigger": "surprise: changed",
                "before": [second.to_dict()],
                "after": [second.to_dict()],
            }
        ],
        final_answer="done",
    )
    recording = _RecordingStreamlit()
    monkeypatch.setattr(itinerary_app, "st", recording)

    itinerary_app._render_run(run.goal, run)

    first_position = next(
        index
        for index, event in enumerate(recording.events)
        if event[0] == "markdown"
        and "<div class='step-card'>" in event[1]
        and "step 10" in event[1]
    )
    revision_position = next(
        index
        for index, event in enumerate(recording.events)
        if event[0] == "markdown" and "Plan revision after Step 10" in event[1]
    )
    second_position = next(
        index
        for index, event in enumerate(recording.events)
        if event[0] == "markdown"
        and "<div class='step-card'>" in event[1]
        and "step 11" in event[1]
    )

    assert first_position < revision_position < second_position
    assert ("expander", "Step 10 details") in recording.events
    assert ("expander", "Step 11 details") in recording.events


def test_revision_html_escapes_trigger_goal_and_tool_hint():
    from itinerary_app import _revision_card_html

    trigger = '<script>alert("trigger")</script>'
    goal = '<script>alert("goal")</script>'
    tool_hint = '<script>alert("hint")</script>'
    rendered = _revision_card_html(
        {
            "after_step": 10,
            "trigger": trigger,
            "before": [{"n": 11, "goal": goal, "tool_hint": tool_hint}],
            "after": [],
        }
    )

    for untrusted in (trigger, goal, tool_hint):
        assert untrusted not in rendered
        assert escape(untrusted) in rendered
    assert "<script>" not in rendered
