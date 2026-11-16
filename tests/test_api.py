"""Tests for the fraud detection API."""

import pytest


def make_payload(v1=0.0, amount=50.0, hour=12):
    """A valid transaction. Only V1 matters to the tiny test model."""
    payload = {f"V{i}": 0.0 for i in range(1, 29)}
    payload["V1"] = v1
    payload["Amount"] = amount
    payload["Hour"] = hour
    return payload


def test_health_reports_model_loaded(client):
    reply = client.get("/health")
    assert reply.status_code == 200
    assert reply.json() == {"status": "ok", "model_loaded": True}


def test_predict_returns_expected_fields(client):
    reply = client.post("/predict", json=make_payload())
    assert reply.status_code == 200
    body = reply.json()
    assert set(body) == {"fraud_probability", "is_fraud", "threshold"}
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["threshold"] == 0.5


def test_fraud_like_transaction_is_flagged(client):
    body = client.post("/predict", json=make_payload(v1=5.0)).json()
    assert body["is_fraud"] is True
    assert body["fraud_probability"] > 0.5


def test_normal_transaction_is_not_flagged(client):
    body = client.post("/predict", json=make_payload(v1=-5.0)).json()
    assert body["is_fraud"] is False
    assert body["fraud_probability"] < 0.5


def test_decision_always_matches_threshold(client):
    for v1 in (-3.0, -1.0, 0.0, 1.0, 2.0, 4.0):
        body = client.post("/predict", json=make_payload(v1=v1)).json()
        assert body["is_fraud"] == (body["fraud_probability"] >= body["threshold"])


def test_same_input_gives_same_output(client):
    first = client.post("/predict", json=make_payload(v1=1.5)).json()
    second = client.post("/predict", json=make_payload(v1=1.5)).json()
    assert first == second


def test_missing_field_is_rejected(client):
    payload = make_payload()
    del payload["V14"]
    assert client.post("/predict", json=payload).status_code == 422


def test_empty_body_is_rejected(client):
    assert client.post("/predict", json={}).status_code == 422


@pytest.mark.parametrize(
    "field, bad_value",
    [
        ("Hour", 24),  # hours go from 0 to 23
        ("Hour", -1),
        ("Hour", 12.5),  # must be a whole number
        ("Amount", -5.0),  # amounts cannot be negative
        ("V3", "not a number"),
    ],
)
def test_invalid_values_are_rejected(client, field, bad_value):
    payload = make_payload()
    payload[field] = bad_value
    assert client.post("/predict", json=payload).status_code == 422