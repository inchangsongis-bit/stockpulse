from unittest.mock import MagicMock

import pytest

from agents.base import extract_claude_text


def fake_response(*blocks):
    resp = MagicMock()
    resp.content = list(blocks)
    return resp


def test_extract_claude_text_returns_plain_text():
    resp = fake_response(MagicMock(type="text", text="hello world"))
    assert extract_claude_text(resp) == "hello world"


def test_extract_claude_text_skips_leading_thinking_block():
    resp = fake_response(
        MagicMock(type="thinking", text=None),
        MagicMock(type="text", text="the actual answer"),
    )
    assert extract_claude_text(resp) == "the actual answer"


def test_extract_claude_text_strips_json_fence():
    resp = fake_response(MagicMock(type="text", text='```json\n{"a": 1}\n```'))
    assert extract_claude_text(resp) == '{"a": 1}'


def test_extract_claude_text_strips_plain_fence():
    resp = fake_response(MagicMock(type="text", text="```\nplain text\n```"))
    assert extract_claude_text(resp) == "plain text"


def test_extract_claude_text_raises_when_no_text_block():
    resp = fake_response(MagicMock(type="thinking", text=None))
    with pytest.raises(ValueError):
        extract_claude_text(resp)
