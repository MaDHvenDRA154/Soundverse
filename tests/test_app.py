"""Preview app API without an LLM: upload, analysis, state, peaks, undo/redo, exports."""
import time

import soundfile as sf
from fastapi.testclient import TestClient

from evals import synth


def test_upload_analyse_export(tmp_path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    from beat_agent.app import server
    monkeypatch.setattr(server, "_llm", {"provider": None, "error": "no key in test", "tried": True})
    c = TestClient(server.app)
    a, _ = synth.make_track(segments=((32, 120, 120),), seed=2)
    p = synth.write(tmp_path / "t.wav", a, 44100, "PCM_24")
    with open(p, "rb") as f:
        sid = c.post("/api/upload", files={"file": ("t.wav", f, "audio/wav")}).json()["id"]
    for _ in range(120):
        st = c.get(f"/api/session/{sid}").json()
        if st["status"] in ("ready", "error"):
            break
        time.sleep(0.25)
    assert st["status"] == "ready", st
    assert len(st["beats"]) == 32 and st["can_undo"] == 0 and not st["llm_available"]
    assert len(c.get(f"/api/session/{sid}/peaks?n=500").json()["peaks"]) == 500
    r = c.post(f"/api/session/{sid}/chat", json={"text": "remove bar 2"}).json()
    assert r["chat"][-1]["status"] == "error" and "unavailable" in r["chat"][-1]["echo"]
    assert c.post(f"/api/session/{sid}/undo").json()["chat"][-1]["status"] == "error"   # nothing to undo
    out = tmp_path / "click.wav"
    out.write_bytes(c.get(f"/api/session/{sid}/export/click").content)
    assert sf.info(out).subtype == "PCM_24"


def test_bad_upload_reports_error(tmp_path):
    from beat_agent.app import server
    c = TestClient(server.app)
    sid = c.post("/api/upload", files={"file": ("x.wav", b"not audio", "audio/wav")}).json()["id"]
    for _ in range(40):
        st = c.get(f"/api/session/{sid}").json()
        if st["status"] in ("ready", "error"):
            break
        time.sleep(0.25)
    assert st["status"] == "error" and st["error"]
