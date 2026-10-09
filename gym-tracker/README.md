# Gym tracker

Watches the camera, notices when someone starts lifting, counts reps and sets,
and reads the weight off the plates. Each finished set is logged locally and,
optionally, appended to the workout-log Google Sheet.

## How it works

1. **Idle:** every `--idle-interval` seconds (default 2), run pose detection on
   one frame. When a person shows up, switch to active.
2. **Active:** MediaPipe Pose on every frame. `features.py` turns landmarks into
   joint angles and hand/elbow heights (in torso lengths, so distance from the
   camera doesn't matter).
3. **Reps:** `reps.py` runs one counter per movement family (squat, hinge,
   press, curl). A rep is leave rest → pass the turn threshold → back to rest,
   then gates check it was really that movement: a squat has the hands up at
   the shoulders, a deadlift has them down by the knees, and so on.
4. **Sets:** `sets.py` groups reps of one movement. A set ends after
   `--rest-gap` seconds (default 25) without a rep. Sets under `--min-reps`
   (default 2) are dropped as noise, since bending to pick up a plate looks
   like one deadlift.
5. **Weight + name:** `weigh.py` sends up to 3 frames from the set, plus the
   rep measurements, to Claude (`claude-opus-5-5`). It names the lift the way
   the log does (Back Squat vs Front Squat, Strict vs Push Press), counts the
   plates, and leaves the weight blank with a note when it can't read them.
6. **Log:** `logbook.py` writes `logs/YYYY-MM-DD.jsonl` plus the frames. When
   nobody has been in frame for `--absent-timeout` (default 60 s), the session
   ends: it prints a summary and, with `--sheet`, appends rows to the sheet
   (back-to-back identical sets collapse into one row, e.g. 3 × 5).

## Setup (Mac)

```bash
cd gym-tracker
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
export ANTHROPIC_API_KEY=...        # for the plate reading
```

The pose model downloads to `models/` on first run.

## Run

```bash
.venv/bin/python gym_tracker.py                    # Pi stream (pibot5-2g.local:8080)
.venv/bin/python gym_tracker.py -s 192.168.4.56    # Pi at an IP
.venv/bin/python gym_tracker.py -s local           # Mac webcam
.venv/bin/python gym_tracker.py -s clip.mp4        # recorded video, every frame
.venv/bin/python gym_tracker.py --sheet            # also append to the workout log
.venv/bin/python gym_tracker.py --no-weigh         # skip Claude (reps/sets only)
```

The Pi side is just the camera stream: `python3 robot_server.py --no-motors` on the Pi.

Tests (synthetic movements, no camera needed): `.venv/bin/python -m pytest tests`.

## Camera placement

- **Side-on to the lifter, i.e. looking at the plate faces.** That's the best
  view for both jobs: knee and hip angles are clearest from the side, and the
  plates can only be read face-on. Edge-on plates (camera in front of the
  lifter) can't be read; the set gets logged with the weight blank.
- Whole body in frame, head to feet, through the bottom of each rep.
- Decent light on the plates. Color-coded bumpers read best.

## Running on the Pi (planned)

The code avoids anything Mac-specific so it can move onto the Pi:

- `-s picamera` reads the camera directly with Picamera2 (stop `robot_server.py`
  first, since only one process can own the camera).
- `--model` defaults to `lite` on ARM, `full` elsewhere.
- Use `--headless`. A systemd unit would make it run whenever the Pi is on.
- **Unverified:** whether `mediapipe==0.10.31` installs on the Pi's
  Debian 13 / Python 3.13 (aarch64). Check that first; the fallback is a
  different mediapipe version or running pose on the Mac/NZXT against the stream.

## Known limits

- **Recognised movements:** squat, hinge (deadlift / trap bar / RDL), overhead
  press (barbell, dumbbell, machine), curl. Bench press, rows, lunges, cleans and
  jerks aren't counted yet. Bench needs a different camera view.
- **Deadlift counting** assumes the last cycle of a set is setting the bar down
  (walking up, gripping and pulling is rep 1; setting it down and standing up
  empty-handed looks like another full cycle). If a set ends any other way the
  count can be one off.
- Thresholds were tuned on synthetic data plus a few public lifting videos,
  not on this gym yet. Expect to adjust `make_counters()` in `reps.py` after the
  first real sessions. The JSONL log keeps each rep's measurements for that.
