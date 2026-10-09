# Gym tracker: status and plan

`README.md` covers how it works and how to run it. This file covers where things
stand, why it's built this way, and what comes next. Started 2026-10-08.

## Goal

The Pi 5 camera watches the lifting area. When someone shows up, it tracks them:
which exercise (overhead press, squat, deadlift, ...), reps, sets, and the weight
from the plates. Each session ends up in the workout-log Google Sheet without
typing anything.

Motors are out of scope. The Pi is just a camera here.

## Decisions

| Decision | Why |
|---|---|
| **Debug on the Mac first, run on the Pi eventually** | Faster iteration with a preview window and the full pose model. The code avoids anything Mac-specific so it can move over (`-s picamera`, `lite` model on ARM, `--headless`). |
| **Pose (MediaPipe) counts reps; Claude names the lift and reads the plates** | Counting reps is a geometry problem that has to run on every frame, locally. The exact lift name (Back vs Front Squat, Strict vs Push Press) and the plate count need judgment and vision, once per set. Pose only decides the movement *family*. Claude gets the frames plus the rep measurements and decides the rest, including leaving the weight blank when it can't read the plates. |
| **Claude via the `claude` CLI, not the API SDK** | Tony's call (2026-10-08). It uses the machine's Claude login, with no API key to manage. The call switches off tools, MCP servers, skills and settings and runs from an empty folder, which took it from about $0.80 to $0.05 of plan usage per set. The cost: the SDK's automatic retry on another model when Claude declines isn't available, so a declined set is just logged without a weight. |
| **Log locally always; the sheet is opt-in (`--sheet`)** | Until the counting is trusted on the real camera, it shouldn't write unattended to the real log. The JSONL log keeps every rep's measurements for tuning. |
| **Row format matches the hand-kept log** | Exercise names follow the sheet's existing ones. Back-to-back identical sets collapse to one row (3 × 5 → Reps 5, Sets 3). Sets is blank for a single set. |
| **Sets under 2 reps are dropped** | Bending to load a plate looks like a single deadlift. |
| **Live feedback on the garage LED clock** | Added 2026-10-09 once the display was up: live rep count during a set, then reps × weight when the set has been read, so a bad read is visible on the spot. |
| **Clock runs AWTRIX NG (unofficial TC002 port)** | Flashed 2026-10-09 at Tony's call: the stock firmware's font had no `?` and its API was undocumented. The TC002 is Linux-based, so official AWTRIX 3 doesn't run on it; [awtrix-ng-tc002](https://github.com/sanderdw/awtrix-ng-tc002) v1.1.2-tc002.4 does, with a documented `/api/v1` API (notifications, screen readback). **Going back:** hold the knob while powering on to boot the stock app, or re-run the installer with `--restore` (it uses `~/.awtrix-ng-tc002/192.168.4.37/20261009-105710/restore-stock.img` on the Mac, which is the clock's own stock partition, so keep that file). The installer needs Google's `adb` (`ADB=~/.awtrix-ng-tc002/platform-tools/adb`); the Homebrew `adb` on the Mac hangs. |
| **Stream at 1280×720** | 640×480 is too small to read plate markings. |

## What's been verified

- **Unit tests:** 8 pass. They use synthetic movement data for each lift, check
  that squat/deadlift/press/curl don't trigger each other, and cover partial
  reps, rest splitting sets, and signal dropouts.
- **Public videos** (Wikimedia Commons):
  - Squat demo: 2 reps, correct.
  - Machine shoulder press: 3 reps, correct. Claude named it "Machine Shoulder
    Press" and left the weight blank because it couldn't read the pin.
  - Army trap-bar deadlift: first counted nothing, because a knee-angle rule
    rejected trap-bar pulls. Fixed. Now 2 of 3 reps in the one continuous shot;
    the video cuts before the bar is set down, which the deadlift rule expects.
  - YouTube-style curl video: only the wide shots count. Close-up crops give
    junk, including one false "press". A fixed camera doesn't have this problem.
- **Plate reading:**
  - Meet deadlift photo captioned 260 kg: read as 255 kg / 562 lb. It counted
    the plates by their markings and noted small change plates might be hidden.
  - Front-on squat photo: correctly declined to guess (plates edge-on).
- **End to end** with the CLI and no API key: video → reps → set → Claude →
  JSONL + frames → session summary.
- **Live Pi stream:** the tracker reads it at about 30 fps. Not tested with a
  person yet; the room was dark.

## First live session (2026-10-09, garage, strict press)

| Try | Reps (actual → counted) | Weight (actual → read) | What changed after |
|---|---|---|---|
| 1 | 5 → 0 | 65 → – | Press lockout gate 145° → 120° (fixed camera angle) |
| 2 | 5 → 4 | 65 → ? | A stray "squat" made the first press look like a misread; frames sent were mid-rep and blurred |
| 3 | 0 → 2 (clearing weights) | – | Claude rejected it once given a `real_set` field |
| 4 | 5 → 5 | 70 → 65 (high) | Change plate taken for a bumper hub → plate inventory in the prompt, 4 at-rest frames |
| 5 | 5 → 5 | 72.5 → 72.5 (medium) | Correct, including the white 1.25 |

## Next steps

- **Bar-on-back squats (open, 2026-10-09):** five back squats under the bar
  counted nothing, and the tracker decided nobody was in frame at 12:56 while
  they were happening. With the bar at squat height, the far plate sits where
  the head and shoulders are, and both the squat signal (shoulder drop) and
  the "is someone here" check depend on the shoulders. Bodyweight squats and
  presses count. Next session's clip + CSV (logs/clips on the Pi, added for
  this) will show what's visible; likely fix is a hip-based squat signal and
  a person check that tolerates hidden shoulders.

0. **Camera placement is fixed** (Tony, 2026-10-09: no flexibility on where it
   goes). It looks along the bar from close to the rack, so the software has to
   cope: plates are read from frames with the bar at rest just before/after a
   set (where "10LB" is legible from here), not from blurred mid-rep frames.
   First live session found the press lockout reads only 126-146° from this
   angle (gate lowered to 120°), and crouching at the laptop produces stray
   single "reps" (now dropped quietly, and the clock only counts from rep 2).
   Next: track the bar/plates themselves so a rep needs the bar to move.
1. **First real session.** Mount the camera side-on to the lifting spot (facing
   the plate faces, whole body in frame, decent light) and run on the Mac with
   the preview window and without `--sheet`.
   Compare the summary with what was actually done.
2. **Tune** the thresholds and gates in `reps.py` (`make_counters()`) using the
   per-rep `rep_stats` in `logs/*.jsonl`. Turn on `--sheet` once the counts are
   right.
3. **Move it onto the Pi:**
   - Check that `mediapipe==0.10.31` installs on Debian 13 / Python 3.13 aarch64
     (unverified; the main risk). If it doesn't, try another mediapipe version,
     or keep pose on the Mac or the NZXT against the stream.
   - Install Claude Code on the Pi and log in (or run `--no-weigh`).
   - Run with `-s picamera --headless` after stopping `robot-stream`, or keep
     the stream service and read `-s localhost`.
   - Add a systemd unit for the tracker, like `robot-stream.service`.
4. **More movements** as needed: bench press (needs a different camera view),
   rows, lunges, cleans and jerks.

## Open questions

- Where exactly the camera goes in the gym, and whether the plates are
  color-coded bumpers (easiest to read) or iron.
- Whether the tracker should notify Tony (e.g. via the Telegram bot) when it logs
  a session, so he can correct a bad read.
