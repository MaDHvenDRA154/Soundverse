"""An editing session: state, atomic turns, echo, undo/redo, export.

A turn is a validated EditPlan. Its ops are applied one after another to a
*copy* of the state, each resolved against the grid as it stands after the
previous op; if any op fails, nothing from the turn is applied. Undo/redo
restores whole turns.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from beat_agent import config
from beat_agent.analysis.schema import BeatGrid
from beat_agent.edit import ops as O
from beat_agent.edit.gridmap import SourceGrid, map_grid
from beat_agent.edit.resolve import EditError, describe, resolve_point, resolve_range
from beat_agent.edit.timeline import Factor, Timeline, render
from beat_agent.io.audio import SourceInfo, iter_blocks, open_source
from beat_agent.io.audition import write_click_audition


@dataclass(frozen=True)
class State:
    timeline: Timeline
    sgrid: SourceGrid
    version: int = 0


@dataclass
class TurnResult:
    status: str                       # applied | clarify | rejected | error | proposal | undo | redo | nothing
    message: str                      # what the user sees (the echo)
    ops: list = field(default_factory=list)          # validated ops (JSON)
    resolved: list = field(default_factory=list)     # per op: what it resolved to
    length_before_s: float | None = None
    length_after_s: float | None = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def fmt_time(s: float) -> str:
    m, x = divmod(s, 60.0)
    return f"{int(m)}:{x:06.3f}"


class EditSession:
    def __init__(self, src: SourceInfo, grid: BeatGrid):
        self.src = src
        self.sr = src.samplerate
        self.state = State(Timeline.whole(src.frames), SourceGrid.from_grid(grid), 0)
        self.undo_stack: list[State] = []
        self.redo_stack: list[State] = []
        self.undo_labels: list[str] = []      # user text of each undoable turn (parallel to undo_stack)
        self.redo_labels: list[str] = []
        self._current_text = ""
        self.pending: dict | None = None      # measured proposal awaiting confirmation
        self.turns: list[dict] = []
        self._grid_cache: dict[int, BeatGrid] = {}
        self._versions = 0

    # ------------------------------------------------------------ views
    def grid(self, state: State | None = None) -> BeatGrid:
        st = state or self.state
        if st.version not in self._grid_cache:
            meta = dict(self.src.to_dict())
            meta.update({"samplerate": self.sr, "edited": st.version > 0})
            self._grid_cache[st.version] = map_grid(st.sgrid, st.timeline, meta, self.history_summary())
        return self._grid_cache[st.version]

    def length_s(self, state: State | None = None) -> float:
        return (state or self.state).timeline.length / self.sr

    def history_summary(self) -> list:
        return [{"user": t["user"], "status": t["status"], "ops": t["ops"]} for t in self.turns]

    def grid_header(self, g: BeatGrid | None = None) -> str:
        g = g or self.grid()
        ts = g.time_signature
        if not g.beats:
            return "Grid: no beats detected."
        unit = {"quarter": "quarter note", "eighth": "eighth note", "dotted_quarter": "dotted quarter"}.get(ts.beat_unit or "", "beat")
        meter = f"{ts.numerator}/{ts.denominator}" if ts.numerator else "meter unknown"
        bpm = f"{g.tempo.bpm:.2f} BPM" if g.tempo.bpm else "tempo unknown"
        head = (f"Grid: {meter}, 1 beat = {unit}, {bpm} ({g.tempo.type}); beats {g.beats[0].index}-{g.beats[-1].index}, "
                f"{f'bars 1-{max(b.bar for b in g.bars)}' if any(b.bar >= 1 for b in g.bars) else 'no bars (meter unknown)'}. "
                "Beats are 0-based, bars 1-based; "
                "beat ranges exclude their end beat, bar ranges include it.")
        amb = self.ambiguity_notes(g)
        return head + ("\n" + "\n".join(amb) if amb else "")

    def ambiguity_notes(self, g: BeatGrid) -> list[str]:
        out = []
        ts = g.time_signature
        if ts.beat_unit == "eighth":
            out.append(f"Beat unit: in {ts.numerator}/{ts.denominator} I count EIGHTH notes ({ts.beats_per_bar} per bar), "
                       "not dotted quarters, so 'beat N' means the N-th eighth note.")
        elif ts.beat_unit == "dotted_quarter":
            out.append(f"Beat unit: in {ts.numerator}/{ts.denominator} I count DOTTED QUARTERS ({ts.beats_per_bar} per bar), "
                       "not eighth notes, so 'beat N' means the N-th dotted quarter.")
        for a in g.tempo.alternatives:
            if a.factor == 2.0 and a.onset_support >= config.OCTAVE_DOUBLE_PLAUSIBLE:
                out.append(f"Tempo octave: counting at {g.tempo.bpm:.2f} BPM; {a.bpm:.2f} BPM is also plausible "
                           f"(half-beat onsets at {a.onset_support:.2f} of on-beat strength). Say 'the tempo is double that' to switch.")
            if a.factor == 0.5 and a.onset_support <= config.OCTAVE_HALF_PLAUSIBLE:
                out.append(f"Tempo octave: counting at {g.tempo.bpm:.2f} BPM; {a.bpm:.2f} BPM is also plausible "
                           f"(alternate beats differ strongly, ratio {a.onset_support:.2f}). Say 'use half tempo' to switch.")
        if g.time_signature.confidence < 0.5 and g.bars:
            out.append(f"Bar lines are uncertain (downbeat confidence {g.time_signature.confidence:.2f}); "
                       "say 'shift the downbeat by one beat' if bar 1 sounds wrong.")
        return out

    # ------------------------------------------------------------ turns
    def apply(self, plan: O.EditPlan, user_text: str = "") -> TurnResult:
        self._current_text = user_text
        res = self._apply(plan)
        self.turns.append({"user": user_text, "reading": plan.reading, "status": res.status,
                           "ops": res.ops, "message": res.message})
        return res

    def _apply(self, plan: O.EditPlan) -> TurnResult:
        ops = plan.ops
        ops_json = [o.model_dump() for o in ops]
        kinds = {type(o) for o in ops}
        before = self.length_s()
        for o in ops:
            if isinstance(o, O.Clarify):
                return TurnResult("clarify", o.question, ops_json)
            if isinstance(o, O.Reject):
                return TurnResult("rejected", o.reason, ops_json)
        if kinds & {O.Undo, O.Redo} and len(ops) > 1:
            return TurnResult("error", "Undo/redo must be on its own; nothing was changed.", ops_json)
        if isinstance(ops[0], O.Undo):
            n = ops[0].turns
            if len(self.undo_stack) < n:
                return TurnResult("error", f"Nothing to undo ({len(self.undo_stack)} edit turn(s) in history).", ops_json)
            names = []
            for _ in range(n):
                self.redo_stack.append(self.state)
                self.state = self.undo_stack.pop()
                lab = self.undo_labels.pop() if self.undo_labels else "?"
                self.redo_labels.append(lab)
                names.append(f"'{lab}'")
            return TurnResult("undo", f"Undid {', '.join(names)}. Length {fmt_time(before)} -> {fmt_time(self.length_s())}.\n"
                              + self.grid_header(), ops_json, length_before_s=before, length_after_s=self.length_s())
        if isinstance(ops[0], O.Redo):
            n = ops[0].turns
            if len(self.redo_stack) < n:
                return TurnResult("error", f"Nothing to redo ({len(self.redo_stack)} available).", ops_json)
            names = []
            for _ in range(n):
                self.undo_stack.append(self.state)
                self.state = self.redo_stack.pop()
                lab = self.redo_labels.pop() if self.redo_labels else "?"
                self.undo_labels.append(lab)
                names.append(f"'{lab}'")
            return TurnResult("redo", f"Redid {', '.join(names)}. Length {fmt_time(before)} -> {fmt_time(self.length_s())}.\n"
                              + self.grid_header(),
                              ops_json, length_before_s=before, length_after_s=self.length_s())
        if any(isinstance(o, O.ProposeCut) for o in ops):
            if len(ops) > 1:
                return TurnResult("error", "A proposal must be on its own; nothing was changed.", ops_json)
            try:
                return self._propose(ops[0], ops_json)
            except EditError as e:
                return TurnResult("error", str(e), ops_json)

        # expand a confirmation into the concrete op it confirms
        concrete = []
        for o in ops:
            if isinstance(o, O.ConfirmProposal):
                if not self.pending:
                    return TurnResult("error", "There is no pending proposal to confirm; nothing was changed.", ops_json)
                if self.pending["version"] != self.state.version:
                    return TurnResult("error", "The track changed since that proposal; ask again so I can re-measure.", ops_json)
                concrete.append(O.Remove(op="remove", range=O.BarRange(unit="bars", **self.pending["bars"])))
            else:
                concrete.append(o)

        # "Cut X and repeat it" selects X (DECISIONS ambiguity 12). A model that emits remove(X) then
        # repeat(X) would delete X and then repeat whatever slid into X's place -- never what was meant.
        notes = []
        for i in range(len(concrete) - 1, 0, -1):
            a, b = concrete[i - 1], concrete[i]
            if isinstance(a, O.Remove) and isinstance(b, O.Repeat) and a.range == b.range:
                del concrete[i - 1]
                r = a.range.model_dump()
                what = f"{r['unit']} {r.get('start')}-{r.get('end')}" if "start" in r else f"the {r['unit']} {r.get('count', '')}"
                notes.append(f"Read 'cut ... and repeat' as selecting {what.strip()}, not deleting it "
                             "(the plan also removed that range, which would have repeated different audio).")

        st = self.state
        lines, resolved = list(notes), []
        try:
            for o in concrete:
                st, line, info = self._apply_op(st, o)
                lines.append(line)
                resolved.append(info)
        except EditError as e:
            return TurnResult("error", f"{e}\nNothing was changed.", ops_json)
        if st.timeline.length < int(0.05 * self.sr):
            return TurnResult("error", "That would leave (almost) no audio; nothing was changed.", ops_json)
        if st.timeline.length > config.MAX_OUTPUT_S * self.sr:
            return TurnResult("error", f"That would make the track longer than {config.MAX_OUTPUT_S / 60:.0f} minutes; "
                              "nothing was changed.", ops_json)
        self.undo_stack.append(self.state)
        self.undo_labels.append(self._current_text or ", ".join(o.op for o in concrete))
        self.redo_stack.clear()
        self.redo_labels.clear()
        self._versions += 1
        self.state = replace(st, version=self._versions)
        self.pending = None
        after = self.length_s()
        msg = "\n".join(lines) + f"\nLength {fmt_time(before)} -> {fmt_time(after)}.\n" + self.grid_header()
        if plan.reading:
            msg = f"Reading: {plan.reading}\n" + msg
        return TurnResult("applied", msg, ops_json, resolved, before, after)

    def _apply_op(self, st: State, o) -> tuple[State, str, dict]:
        # intermediate states inside a turn share the old version number, so only
        # the committed state may use the cache
        g = self.grid() if st is self.state else map_grid(st.sgrid, st.timeline, {"samplerate": self.sr}, [])
        sr, total = self.sr, st.timeline.length
        tl = st.timeline
        if isinstance(o, O.GRID_OPS):
            try:
                if isinstance(o, O.ShiftDownbeat):
                    sg = st.sgrid.shift_downbeat(o.beats)
                    line = f"Grid: moved every bar line {o.beats:+d} beat(s); audio unchanged."
                elif isinstance(o, O.SetBeatsPerBar):
                    sg = st.sgrid.set_beats_per_bar(o.beats_per_bar)
                    line = f"Grid: now {o.beats_per_bar} beats per bar, bar 1 still starts at beat 0; audio unchanged."
                else:
                    sg = st.sgrid.set_octave(o.factor)
                    line = (f"Grid: beat level x{o.factor:g} ({'a beat inserted halfway between each pair' if o.factor == 2 else 'every other beat kept'}); "
                            f"bar lines stay where they were, so each bar now has {sg.beats_per_bar} beats; audio unchanged.")
            except ValueError as e:
                raise EditError(str(e))
            st2 = State(tl, sg, st.version)
            g2 = map_grid(sg, tl, {"samplerate": sr}, [])
            line += f" Now {g2.tempo.bpm} BPM, {len(g2.beats)} beats, {g2.n_bars()} bars."
            return st2, line, {"op": o.op}

        span = resolve_range(o.range, g, sr, total)
        a, b = span.a, span.b
        if b - a <= 0:
            raise EditError(f"{span.what} is empty.")
        where = describe(g, a, b, sr)
        notes = (" " + " ".join(span.notes)) if span.notes else ""
        info = {"op": o.op, "addressed": span.what, "output_samples": [a, b], "seconds": [a / sr, b / sr]}
        if isinstance(o, O.Remove):
            if a == 0 and b >= total:
                raise EditError("That would remove the whole track.")
            tl2 = tl.remove(a, b)
            line = f"Removed {span.what}: {where}.{notes}"
        elif isinstance(o, O.KeepOnly):
            tl2 = tl.keep_only(a, b)
            line = f"Kept only {span.what}: {where}; everything else removed.{notes}"
        elif isinstance(o, O.Repeat):
            if o.times_total > config.MAX_REPEAT_TOTAL:
                raise EditError(f"Repeating {o.times_total} times is more than the {config.MAX_REPEAT_TOTAL} I allow in one go.")
            tl2 = tl.repeat(a, b, o.times_total)
            line = (f"Repeated {span.what} so it plays {o.times_total} times in total, back to back in place "
                    f"(original + {o.times_total - 1} cop{'y' if o.times_total == 2 else 'ies'}): {where}; "
                    f"the copies occupy {b / sr:.3f}-{(b + (o.times_total - 1) * (b - a)) / sr:.3f} s.{notes} "
                    f"(If you meant the original plus {o.times_total} more, that is {o.times_total + 1} plays: "
                    f"say 'play it {o.times_total + 1} times in total'.)")
        elif isinstance(o, O.Move):
            p, pdesc = resolve_point(o.to, g, sr, total)
            if a < p < b:
                raise EditError(f"Cannot move {span.what} to a point inside itself ({pdesc}).")
            tl2 = tl.move(a, b, p)
            line = f"Moved {span.what} ({where}) to {pdesc} ({p / sr:.3f} s in the track before the move).{notes}"
        elif isinstance(o, O.Insert):
            p, pdesc = resolve_point(o.at, g, sr, total)
            tl2 = tl.insert_copy(a, b, p, o.copies)
            line = f"Inserted {o.copies} cop{'y' if o.copies == 1 else 'ies'} of {span.what} ({where}) {pdesc} ({p / sr:.3f} s).{notes}"
        elif isinstance(o, O.Fade):
            tl2 = tl.apply_factor(a, b, Factor(f"fade_{o.direction}", o.shape))
            line = f"Fade {o.direction} ({o.shape.replace('_', '-')}) over {span.what}: {where}.{notes}"
        elif isinstance(o, O.Gain):
            tl2 = tl.apply_factor(a, b, Factor("gain", value=float(10 ** (o.db / 20))))
            line = f"Gain {o.db:+.1f} dB on {span.what}: {where}.{notes}"
        else:
            raise EditError(f"Unsupported op {o.op}")
        return State(tl2, st.sgrid, st.version), line, info

    # ------------------------------------------------------------ proposals
    def _propose(self, o: O.ProposeCut, ops_json) -> TurnResult:
        g = self.grid()
        bars = [b for b in g.bars if b.bar >= 1 and b.complete]
        if len(bars) < 4:
            raise EditError("'Boring' is not something I can measure, and this track has too few bars for me to "
                            "propose a measured candidate. Tell me which beats or bars to cut.")
        W = o.bars or min(config.PROPOSE_CUT_BARS, max(1, len(bars) // 4))
        if W >= len(bars):
            raise EditError(f"A {W}-bar stretch would be the whole track ({len(bars)} bars).")
        tmp = Path(self.render_current())
        src = open_source(tmp)
        y = np.concatenate([blk.mean(axis=1) for blk in iter_blocks(src)])
        sr = src.samplerate
        db = np.array([10 * np.log10(np.mean(y[int(b.start * sr):int(b.end * sr)] ** 2) + 1e-12) for b in bars])
        best, best_v = None, np.inf
        for i in range(len(bars) - W + 1):
            if bars[i + W - 1].bar - bars[i].bar != W - 1:
                continue    # not contiguous
            v = db[i:i + W].mean()
            if v < best_v:
                best, best_v = i, v
        if best is None:
            raise EditError("No contiguous stretch of complete bars to propose.")
        b0, b1 = bars[best], bars[best + W - 1]
        track_db = 10 * np.log10(np.mean(y ** 2) + 1e-12)
        self.pending = {"bars": {"start": b0.bar, "end": b1.bar}, "version": self.state.version,
                        "criterion": "lowest mean RMS level over a contiguous stretch of complete bars",
                        "mean_db": round(float(best_v), 2), "track_db": round(float(track_db), 2)}
        where = describe(g, int(round(b0.start * self.sr)), int(round(b1.end * self.sr)), self.sr)
        contrast = float(track_db - best_v)
        self.pending["contrast_db"] = round(contrast, 2)
        length_note = "your choice" if o.bars else "my default: an 8-bar phrase, or a quarter of a short track"
        if contrast < config.PROPOSE_MIN_CONTRAST_DB:
            msg = ("'Boring' is not something I can measure, so I have not cut anything.\n"
                   f"I also have no strong measured candidate: the quietest {W}-bar stretch, bars {b0.bar}-{b1.bar} "
                   f"({where}), is only {contrast:.1f} dB below the track average ({best_v:.1f} vs {track_db:.1f} dBFS), "
                   f"under the {config.PROPOSE_MIN_CONTRAST_DB:.0f} dB I would call a clear difference "
                   f"(length {W} bars = {length_note}).\n"
                   "Tell me which bars you find boring, or reply 'yes, cut it' if you do want exactly those bars removed.\n"
                   + self.grid_header())
        else:
            msg = ("'Boring' is not something I can measure, so I have not cut anything.\n"
                   f"Measured candidate: bars {b0.bar}-{b1.bar} ({where}) are the quietest {W}-bar stretch, "
                   f"{contrast:.1f} dB below the track average ({best_v:.1f} vs {track_db:.1f} dBFS; criterion: lowest "
                   f"mean RMS level; length {W} bars = {length_note}).\n"
                   "Reply 'yes, cut it' to remove exactly those bars, or tell me which bars you mean.\n" + self.grid_header())
        return TurnResult("proposal", msg, ops_json, [self.pending])

    # ------------------------------------------------------------ output
    def render_current(self, out_path: str | None = None) -> str:
        p = Path(out_path) if out_path else Path(config_cache()) / f"session_{id(self)}_v{self.state.version}.wav"
        if out_path or not p.exists():
            render(self.src, self.state.timeline, p)
        return str(p)

    def export(self, out_dir, stem: str) -> dict:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        audio = out_dir / f"{stem}.wav"
        r = render(self.src, self.state.timeline, audio)
        g = self.grid()
        (out_dir / f"{stem}.grid.json").write_text(g.model_dump_json(indent=1))
        edited_src = open_source(audio)
        aud = write_click_audition(edited_src, g, out_dir / f"{stem}.click.wav")
        return {"audio": str(audio), "grid": str(out_dir / f"{stem}.grid.json"), "click": aud["path"],
                "splices_s": [s / self.sr for s in r["splices"]], "splice_points": self.state.timeline.splice_points(),
                "clipped_samples": r["clipped_samples"]}


def config_cache() -> str:
    from beat_agent.io.audio import cache_dir
    d = cache_dir() / "renders"
    d.mkdir(parents=True, exist_ok=True)
    return str(d)
