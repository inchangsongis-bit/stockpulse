from unittest.mock import MagicMock

from analysis.finbert_sentiment import score_text


def _mock_pipeline(monkeypatch, label, score):
    fake_pipeline = MagicMock(return_value=[{"label": label, "score": score}])
    monkeypatch.setattr("analysis.finbert_sentiment._get_pipeline", lambda: fake_pipeline)
    return fake_pipeline


def test_score_text_maps_positive_label_to_positive_sentiment(monkeypatch):
    _mock_pipeline(monkeypatch, "positive", 0.9)
    sentiment, label, confidence = score_text("great earnings beat")
    assert sentiment == 0.9
    assert label == "positive"
    assert confidence == 0.9


def test_score_text_maps_negative_label_to_negative_sentiment(monkeypatch):
    _mock_pipeline(monkeypatch, "negative", 0.8)
    sentiment, label, confidence = score_text("stock plunges on weak guidance")
    assert sentiment == -0.8
    assert label == "negative"


def test_score_text_maps_neutral_label_to_zero_sentiment(monkeypatch):
    _mock_pipeline(monkeypatch, "neutral", 0.7)
    sentiment, label, confidence = score_text("company held its annual meeting")
    assert sentiment == 0.0
    assert label == "neutral"


def test_score_text_truncates_long_input(monkeypatch):
    fake_pipeline = _mock_pipeline(monkeypatch, "neutral", 0.5)
    score_text("x" * 5000)
    called_with = fake_pipeline.call_args[0][0]
    assert len(called_with) == 2000
