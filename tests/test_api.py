from fastapi.testclient import TestClient

from backend.api import create_app
from backend.conversation import SessionStore, dialogue_prompt
from backend.model_manager import ModelManager


def test_persisted_multiturn_session(tiny, tmp_path):
    model, vocabulary = tiny
    manager = ModelManager(model.parent, vocabulary)
    file = tmp_path / "state.sqlite3"
    client = TestClient(create_app(manager, SessionStore(file)))
    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    spec = client.get("/api/config").json()
    assert spec["models"][0]["task"] == "dialogue"
    session = client.post("/api/sessions", json={"model": model.name}).json()
    url = "/api/sessions/" + session["id"]
    for message in ("hello", "hello again"):
        payload = {"text": message, "revision": session["revision"], "backend": "cpu"}
        response = client.post(url + "/messages", json=payload)
        assert response.status_code == 200
        assert response.json()["stats"]["milliseconds"]["neural_decoder"] > 0
        assert client.post(url + "/messages", json=payload).status_code == 409
        session = response.json()["session"]
    reloaded = TestClient(create_app(manager, SessionStore(file)))
    assert reloaded.get(url).json() == session
    assert len(session["turns"]) == 2
    reset = reloaded.post(url + "/reset", json={"revision": session["revision"]})
    assert reset.json()["turns"] == []
    assert reset.json()["revision"] == session["revision"] + 1


def test_dialogue_prompt_includes_prior_assistant_response():
    assert dialogue_prompt([{"user": "hello", "assistant": "hi"}], "again") == (
        "User: hello\nAssistant: hi\nUser: again\nAssistant:"
    )
