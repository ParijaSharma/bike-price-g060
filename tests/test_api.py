from fastapi.testclient import TestClient

import app.main as m

GOOD = dict(age_years=4, km_driven=30000, engine_cc=150, owners=1,
            service_records=5, accidents=0, brand_tier="mid", city_zone="North")


class DummyModel:
    def predict(self, X):
        return [55.0]


client = TestClient(m.app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


def test_predict_ok(monkeypatch):
    monkeypatch.setattr(m, "_model", DummyModel())
    r = client.post("/predict", json=GOOD)
    assert r.status_code == 200 and r.json()["prediction"] == 55.0


def test_rejects_service_records_above_35():
    assert client.post("/predict", json={**GOOD, "service_records": 36}).status_code == 422


def test_rejects_km_below_500():
    assert client.post("/predict", json={**GOOD, "km_driven": 499}).status_code == 422