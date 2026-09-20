"""The prompt must be identical across conditions except for how the sign is presented."""

import json

from vlm_parking.prompts import TASK, Answer, build_prompt, messages, render_query, render_sign
from vlm_parking.schema import DAYS, Panel, Query, Sign

ALL = list(DAYS)
SIGN = Sign(sign_id="t", panels=[
    Panel(order=0, rule="no_parking", days=["THU"], start="08:00", end="10:00"),
    Panel(order=1, rule="no_parking", days=ALL, start="18:00", end="08:00", district="31", tow_away=True),
    Panel(order=2, rule="time_limited", days=ALL[:6], start="08:00", end="18:00", limit_min=120, district="31"),
])
Q = Query(day="THU", time="09:00", duration_min=90, permit_district="31")


def test_conditions_differ_only_in_the_sign_part():
    a, b = build_prompt(Q), build_prompt(Q, SIGN)
    assert a.startswith(TASK) and b.startswith(TASK)
    assert render_query(Q) in a and render_query(Q) in b
    assert "The sign is shown in the image." in a
    assert a.replace("The sign is shown in the image.", render_sign(SIGN)) == b


def test_query_wording():
    text = render_query(Q)
    assert "Thursday at 09:00" in text and "1 hour 30 minutes" in text and "district 31" in text
    assert "no parking permit" in render_query(Query(day="MON", time="10:00", duration_min=60))


def test_sign_rendering_covers_every_field():
    text = render_sign(SIGN)
    assert "Panel 0: No parking; 08:00-10:00; Thursday" in text
    assert "overnight, into the next day" in text  # 18:00-08:00
    assert "district 31 permit holders exempt" in text and "tow-away" in text
    assert "limit 120 minutes" in text
    assert "every day" in text and "Saturday" in text


def test_all_day_and_permit_only_rendering():
    sign = Sign(sign_id="t", panels=[
        Panel(order=0, rule="no_stopping", days=ALL),
        Panel(order=1, rule="permit_only", days=ALL, district="7"),
    ])
    text = render_sign(sign)
    assert "No stopping; all day; every day" in text
    assert "district 7 permit holders required" in text


def test_messages_put_the_image_first():
    content = messages(Q)[0]["content"]
    assert content[0]["type"] == "image" and content[1]["type"] == "text"
    assert [c["type"] for c in messages(Q, SIGN)[0]["content"]] == ["text"]


def test_answer_schema_order_and_validation():
    assert list(Answer.model_json_schema()["properties"]) == ["reason", "governing_panel", "verdict"]
    a = Answer.model_validate_json('{"reason": "Street cleaning applies.", "governing_panel": 0, "verdict": "illegal"}')
    assert a.verdict == "illegal" and a.governing_panel == 0
    for bad in ['{"reason": "x", "governing_panel": 0, "verdict": "maybe"}', '{"verdict": "legal"}', "not json"]:
        try:
            Answer.model_validate_json(bad)
            raise AssertionError(f"should have failed: {bad}")
        except Exception as err:
            assert "AssertionError" not in type(err).__name__


def test_prompt_is_deterministic():
    assert build_prompt(Q, SIGN) == build_prompt(Q, SIGN)
    assert json.dumps(Answer.model_json_schema(), sort_keys=True) == json.dumps(Answer.model_json_schema(), sort_keys=True)
