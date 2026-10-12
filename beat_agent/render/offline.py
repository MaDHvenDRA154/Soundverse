"""Render a recreation page to WAV in a headless browser (OfflineAudioContext).

Uses Playwright's Chromium; falls back to an installed Chrome/Edge if the
bundled browser is not present. The page's own synthesis code is what runs:
`window.beatAgent.renderOffline(loops)` builds the same voices offline.
"""
from __future__ import annotations

import base64
from pathlib import Path


class OfflineRenderError(RuntimeError):
    pass


def render_offline(html_path, wav_path, loops: int = 1, sample_rate: int = 44100, timeout_ms: int = 120000) -> str:
    from playwright.sync_api import sync_playwright
    url = Path(html_path).resolve().as_uri()
    with sync_playwright() as p:
        browser = None
        errors = []
        for kw in ({}, {"channel": "chrome"}, {"channel": "msedge"}):
            try:
                browser = p.chromium.launch(headless=True, **kw)
                break
            except Exception as e:  # browser flavour not installed
                errors.append(str(e).splitlines()[0])
        if browser is None:
            raise OfflineRenderError("No Chromium available for offline render: " + " | ".join(errors))
        try:
            page = browser.new_page()
            requests = []
            page.on("request", lambda r: requests.append(r.url) if not r.url.startswith("file:") else None)
            page.goto(url)
            page.wait_for_function("() => window.beatAgent !== undefined", timeout=timeout_ms)
            b64 = page.evaluate("([l, sr]) => window.beatAgent.renderOffline(l, sr)", [loops, sample_rate])
            if requests:
                raise OfflineRenderError(f"page made network requests: {requests[:3]}")
        finally:
            browser.close()
    Path(wav_path).parent.mkdir(parents=True, exist_ok=True)
    Path(wav_path).write_bytes(base64.b64decode(b64))
    return str(wav_path)
