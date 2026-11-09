import argparse
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend.conversation import SessionStore, dialogue_prompt
from backend.model_manager import ModelManager
from ncm_engine.device import supported_backends

LOG = logging.getLogger(__name__)
FRONTEND = Path(__file__).resolve().parents[1] / "frontend"


class NewSession(BaseModel):
    model: str = Field(min_length=1, max_length=128)


class Message(BaseModel):
    text: str = Field(min_length=1, max_length=30000)
    revision: int = Field(ge=0)
    backend: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    profile: Literal["quality", "balanced", "ultra"] = "balanced"
    context_limit: int | None = Field(default=None, ge=1)
    history_turns: int = Field(default=8, ge=0, le=24)
    temperature: float | None = Field(default=None, ge=0, le=10)
    top_k: int | None = Field(default=None, ge=0)
    top_p: float | None = Field(default=None, ge=0, le=1)
    cfg_scale: float | None = Field(default=None, ge=0, le=8)


class Reset(BaseModel):
    revision: int = Field(ge=0)


def create_app(manager, store):
    app = FastAPI(title="NCM Chat")

    def model(name):
        if name not in manager.models:
            raise HTTPException(404, "Model is unavailable")
        return manager.models[name]

    def current(session_id, revision=None):
        try:
            session = store.get(session_id)
        except KeyError as error:
            raise HTTPException(404, "Session not found") from error
        if revision is not None and session["revision"] != revision:
            raise HTTPException(409, "Session changed. Reload it and retry.")
        return session

    def replace(session, turns):
        updated = store.replace(session, turns)
        if updated is None:
            raise HTTPException(409, "Session changed. Reload it and retry.")
        return updated

    @app.get("/api/config")
    def config():
        models = []
        for spec in manager.models.values():
            models.append(
                {
                    k: v
                    for k, v in asdict(spec).items()
                    if k not in {"directory", "vocabulary"}
                }
            )
        return {"models": models, "backends": ["auto", *supported_backends()]}

    @app.post("/api/sessions", status_code=201)
    def create(request: NewSession):
        model(request.model)
        return store.create(request.model)

    @app.get("/api/sessions/{session_id}")
    def load(session_id: str):
        return current(session_id)

    @app.post("/api/sessions/{session_id}/reset")
    def reset(session_id: str, request: Reset):
        return replace(current(session_id, request.revision), [])

    @app.post("/api/sessions/{session_id}/messages")
    def send(session_id: str, request: Message):
        session = current(session_id, request.revision)
        spec = model(session["model"])
        text = request.text.strip()
        if not text:
            raise HTTPException(422, "Enter some text")
        if len(session["turns"]) >= 250:
            raise HTTPException(409, "Reset the session or start a new one")
        history = (
            session["turns"][-request.history_turns :] if request.history_turns else []
        )
        prompt = dialogue_prompt(history, text) if spec.task == "dialogue" else text
        sampling = request.model_dump(
            include={"temperature", "top_k", "top_p", "cfg_scale"}, exclude_none=True
        )
        try:
            output, stats = manager.generate(
                spec.name,
                request.backend,
                request.profile,
                prompt,
                sampling,
                request.context_limit,
            )
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        except (RuntimeError, OSError) as error:
            LOG.exception("Inference failed")
            raise HTTPException(
                503, "Inference failed. Check the server log."
            ) from error
        turns = [*session["turns"], {"user": text, "assistant": output}]
        return {"session": replace(session, turns), "stats": stats}

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(FRONTEND / "index.html")

    app.mount("/static", StaticFiles(directory=FRONTEND), name="static")
    return app


def main():
    parser = argparse.ArgumentParser(
        description="Launch the offline NCM web app on loopback."
    )
    parser.add_argument("--models", type=Path, default=Path("artifacts"))
    parser.add_argument("--vocabulary", type=Path, default=Path("artifacts/gpt2"))
    parser.add_argument("--sessions", type=Path, default=Path("sessions/chat.sqlite3"))
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    app = create_app(
        ModelManager(args.models, args.vocabulary), SessionStore(args.sessions)
    )
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
