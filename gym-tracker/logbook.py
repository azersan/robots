"""
Record finished sets: a local JSONL log (always), and optionally rows in
Tony's workout-log Google Sheet (via the `gws` CLI) at the end of a session.

Sheet columns: Date | Exercise | Weight (lb) | Reps | Sets | Notes.
Back-to-back sets of the same exercise, weight, and reps collapse into one
row ("3 sets of 5" -> Reps 5, Sets 3), matching how the log is kept by hand.
"""

import datetime
import json
import os
import subprocess

SHEET_ID = "1XVxBA799-1dJff8g3gJjNtUPHUZrQHU-8ZKt7iHs-uU"


class Logbook:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(os.path.join(log_dir, "frames"), exist_ok=True)
        self.session = []   # records for the sheet, cleared each session

    def record(self, lift_set, reading=None, error=None):
        """Write one finished set. `reading` is a weigh.SetReading or None."""
        started = datetime.datetime.now()
        stamp = started.strftime("%Y%m%d-%H%M%S")
        frames = []
        for i, jpg in enumerate(lift_set.keyframes):
            path = os.path.join(self.log_dir, "frames", f"{stamp}-set{lift_set.id}-{i}.jpg")
            with open(path, "wb") as f:
                f.write(jpg)
            frames.append(path)
        for i, (label, jpg) in enumerate(lift_set.context):
            path = os.path.join(self.log_dir, "frames", f"{stamp}-set{lift_set.id}-ctx{i}.jpg")
            with open(path, "wb") as f:
                f.write(jpg)
            frames.append(path)

        rec = {
            "date": started.strftime("%Y-%m-%d"),
            "time": started.strftime("%H:%M:%S"),
            "movement": lift_set.movement,
            "reps": lift_set.reps,
            "cycles": len(lift_set.events),
            "duration_s": round(lift_set.t_end - lift_set.t_start, 1),
            "exercise": reading.exercise if reading else lift_set.movement,
            "weight_lb": reading.total_weight_lb if reading else None,
            "reading": reading.model_dump() if reading else None,
            "error": error,
            # Claude judged it not a real set (stray movement): kept here for
            # tuning, left out of the session summary and the sheet.
            "rejected": bool(reading) and not reading.real_set,
            "frames": frames,
            # Per-rep measurements, for tuning the thresholds in reps.py.
            "rep_stats": [{k: round(v, 3) for k, v in e.stats.items()} for e in lift_set.events],
        }
        path = os.path.join(self.log_dir, started.strftime("%Y-%m-%d") + ".jsonl")
        with open(path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if not rec["rejected"]:
            self.session.append(rec)
        return rec

    def session_rows(self):
        """Collapse this session's sets into sheet rows."""
        rows = []
        for rec in self.session:
            weight = rec["weight_lb"]
            weight = "" if weight is None else f"{weight:g}"
            reading = rec["reading"] or {}
            notes = ["auto-logged"]
            if not reading:
                notes.append("weight not read")
            elif reading.get("confidence") != "high":
                notes.append(f"{reading.get('confidence')} confidence")
            key = (rec["date"], rec["exercise"], weight, str(rec["reps"]))
            if rows and tuple(rows[-1][:4]) == key:
                rows[-1][4] = str(int(rows[-1][4]) + 1)
            else:
                rows.append([*key, "1", ", ".join(notes)])
        for r in rows:
            if r[4] == "1":
                r[4] = ""   # the log leaves Sets blank for a single set
        return rows

    def push_session(self, to_sheet):
        """End of session: print the summary and, if asked, append to the sheet."""
        rows = self.session_rows()
        self.session = []
        if not rows:
            return
        print("\nSession summary:")
        for r in rows:
            print(f"  {r[1]:<16} {r[2] or '?':>6} lb  {r[4] or 1} x {r[3]}   ({r[5]})")
        if not to_sheet:
            return
        result = subprocess.run(
            ["gws", "sheets", "+append", "--spreadsheet", SHEET_ID, "--json-values", json.dumps(rows)],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            print(f"Appended {len(rows)} row(s) to the workout log.")
        else:
            print(f"Sheet append failed (rows are still in {self.log_dir}):\n{result.stderr.strip()}")
