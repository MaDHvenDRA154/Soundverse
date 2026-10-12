"""Preview app: upload any audio, see the grid, chat-edit it, undo/redo, export.

    uvicorn beat_agent.app.server:app --port 8000
"""
from __future__ import annotations

import threading
import traceback
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from beat_agent.edit import ops as O
from beat_agent.io.audio import AudioInputError, cache_dir, iter_blocks, open_source

app = FastAPI(title="Beat Agent")
STATIC = Path(__file__).parent / "static"
SESSIONS: dict[str, dict] = {}
_llm = {"provider": None, "error": None, "tried": False}
_lock = threading.Lock()


def llm():
    with _lock:
        if not _llm["tried"]:
            _llm["tried"] = True
            try:
                from beat_agent.agent.llm import GeminiProvider
                _llm["provider"] = GeminiProvider()
            except Exception as e:
                _llm["error"] = str(e)
        return _llm["provider"]


def _get(sid: str) -> dict:
    s = SESSIONS.get(sid)
    if not s:
        raise HTTPException(404, "unknown session")
    return s


def _analyse(sid: str, path: Path):
    s = SESSIONS[sid]
    try:
        from beat_agent.analysis.pipeline import analyze
        from beat_agent.edit.session import EditSession
        src = open_source(path)
        s["status"] = "analysing"
        grid = analyze(path, src)
        s["session"] = EditSession(src, grid)
        p = llm()
        if p is not None:
            from beat_agent.agent.agent import Agent
            s["agent"] = Agent(p, s["session"])
        s["status"] = "ready"
        s["analysis_s"] = grid.analysis.get("timings", {}).get("total_s")
    except AudioInputError as e:
        s.update(status="error", error=str(e))
    except Exception as e:  # never leave the UI hanging
        s.update(status="error", error=f"analysis failed: {e}", trace=traceback.format_exc(limit=4))


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    sid = uuid.uuid4().hex[:12]
    d = cache_dir() / "uploads" / sid
    d.mkdir(parents=True, exist_ok=True)
    name = Path(file.filename or "audio").name
    p = d / name
    p.write_bytes(await file.read())
    SESSIONS[sid] = {"status": "decoding", "name": name, "path": str(p), "chat": []}
    threading.Thread(target=_analyse, args=(sid, p), daemon=True).start()
    return {"id": sid}


def _state(s: dict) -> dict:
    out = {"status": s["status"], "name": s["name"], "error": s.get("error"), "chat": s["chat"],
           "llm_available": llm() is not None, "llm_error": _llm["error"]}
    sess = s.get("session")
    if sess is None:
        return out
    g = sess.grid()
    out.update({
        "version": sess.state.version, "duration": g.duration, "header": sess.grid_header(g),
        "beats": [[round(b.time, 4), b.index, b.bar, b.beat_in_bar] for b in g.beats],
        "tempo_changes": g.tempo.changes, "splices": g.analysis.get("splices_s", []),
        "messages": g.messages, "can_undo": len(sess.undo_stack), "can_redo": len(sess.redo_stack),
        "usage": s["agent"].ledger.to_dict() if s.get("agent") else None,
        "source": {k: g.source.get(k) for k in ("samplerate", "channels", "export_subtype", "input_format")},
        "confidence": sess.state.sgrid.base.confidence.model_dump(exclude={"how"}),
        "analysis_s": s.get("analysis_s"),
    })
    return out


@app.get("/api/session/{sid}")
def state(sid: str):
    return _state(_get(sid))


def _current_wav(s: dict) -> str:
    return s["session"].render_current()


@app.get("/api/session/{sid}/peaks")
def peaks(sid: str, n: int = 1600):
    s = _get(sid)
    if s.get("session") is None:
        raise HTTPException(409, "not ready")
    src = open_source(_current_wav(s))
    y = np.concatenate([b.mean(axis=1) for b in iter_blocks(src)]) if src.frames else np.zeros(1)
    n = max(100, min(4000, n))
    edges = np.linspace(0, len(y), n + 1).astype(int)
    mx = [float(np.abs(y[a:b]).max()) if b > a else 0.0 for a, b in zip(edges[:-1], edges[1:])]
    return {"peaks": [round(v, 4) for v in mx], "duration": src.duration}


@app.get("/api/session/{sid}/audio")
def audio(sid: str):
    s = _get(sid)
    return FileResponse(_current_wav(s), media_type="audio/wav", filename=f"edited_{Path(s['name']).stem}.wav")


@app.get("/api/session/{sid}/export/{what}")
def export(sid: str, what: str):
    s = _get(sid)
    sess = s["session"]
    d = cache_dir() / "exports" / sid / f"v{sess.state.version}"
    if not (d / "edit.wav").exists():
        sess.export(d, "edit")
    f = {"audio": "edit.wav", "grid": "edit.grid.json", "click": "edit.click.wav"}.get(what)
    if not f:
        raise HTTPException(404)
    return FileResponse(d / f, filename=f"{Path(s['name']).stem}_{f}")


class Chat(BaseModel):
    text: str


@app.post("/api/session/{sid}/chat")
def chat(sid: str, body: Chat):
    s = _get(sid)
    if s.get("session") is None:
        raise HTTPException(409, "analysis not finished")
    text = body.text.strip()
    if not text:
        raise HTTPException(400, "empty message")
    if s.get("agent") is None:
        entry = {"user": text, "status": "error", "echo": f"Chat is unavailable: {_llm['error']}. Undo/redo and export still work."}
    else:
        res, tr = s["agent"].handle(text)
        entry = {"user": text, "status": res.status, "echo": res.message, "ops": tr.plan["ops"] if tr.plan else None}
    s["chat"].append(entry)
    return _state(s)


@app.post("/api/session/{sid}/{action}")
def undo_redo(sid: str, action: str):
    if action not in ("undo", "redo"):
        raise HTTPException(404)
    s = _get(sid)
    sess = s["session"]
    plan = O.EditPlan.model_validate({"reading": f"{action} button", "ops": [{"op": action}]})
    res = sess.apply(plan, f"[{action} button]")
    s["chat"].append({"user": f"[{action}]", "status": res.status, "echo": res.message, "ops": [{"op": action}]})
    return _state(s)
