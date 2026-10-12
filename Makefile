# One-command entry points. Inside Docker:  docker compose run --rm app make eval
PY ?= uv run python
AUDIO ?= testpack
MANIFEST ?= evals/manifest.yaml

.PHONY: setup test test-fast eval eval-robustness eval-part1 eval-part2 eval-part3 status demo app clean-evals

setup:
	uv sync --extra dev
	uv run playwright install chromium

test-fast:            ## unit tests + no-hardcoding guard + synthetic generalisation (no 12-min clip)
	$(PY) -m pytest -q -m "not slow and not llm"

test:                 ## everything, including the 12-minute clip and neural-model cases
	$(PY) -m pytest -q -m "not llm"

eval: clean-evals eval-robustness eval-part1 eval-part2 eval-part3 status   ## regenerate every file under evals/out and evals/results, then the README status block

eval-part1:
	$(PY) -m evals.run part1 --manifest $(MANIFEST)
eval-part2:
	$(PY) -m evals.run part2 --manifest $(MANIFEST)
eval-part3:
	$(PY) -m evals.run part3 --manifest $(MANIFEST)

demo:
	$(PY) -m evals.run demo --manifest $(MANIFEST)

app:
	$(PY) -m uvicorn beat_agent.app.server:app --port 8000

clean-evals:
	$(PY) -c "import shutil; [shutil.rmtree(p, ignore_errors=True) for p in ('evals/out', 'evals/results')]"

eval-robustness:
	$(PY) -m evals.robustness --slow
status:
	$(PY) -m evals.status
