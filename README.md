# Libras Recognizer (demo)

A live demo that opens the webcam in the browser, recognizes **Libras** signs and
lets anyone **teach a new sign on the spot** ("type `J`, record, done").

All the logic is in **Python** (FastAPI + MediaPipe + scikit-learn). The browser
only shows the camera and draws the results. Everything runs locally: no internet
is needed during the demo and no video leaves the laptop.

---

## 1. Run it

**One command** (macOS / Linux):

```bash
./run.sh
```

Wait for the line `READY -> open http://localhost:8000 in Chrome` (the very first
start can take up to a minute while macOS checks the libraries; don't press Ctrl+C),
then open **http://localhost:8000** in **Chrome** and allow the camera.
You don't need to activate the virtualenv for `./run.sh`.

The first time, `run.sh` creates a virtual environment (`.venv/`) and installs the
dependencies. On the first start the app also downloads the MediaPipe model
(13 MB, into `models/`), so **the first run needs internet**. After that it works
offline. Stop the server with `Ctrl+C`.

**Manual way** (same thing, step by step):

```bash
python3 -m venv .venv                        # Python 3.11 – 3.14
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --port 8000
```

> Chrome only allows the camera on `localhost` or HTTPS, so open the page on the
> same laptop that runs Python.

---

## 2. Using the app

### Recognize (default)

* Raise your hand **above the dashed line**, make the sign, then **hold still**
  for a moment or **lower your hands**.
* The app shows the sign in big letters, its confidence and the 2 next best guesses.
  If it is not sure, it says **"Not recognized, try again"**.
* **Spelling mode** (`S` key): only single letters are considered and they join
  into a word (`J U L I A`). Lower your hands for 1.5 s to start a new word.
* **Pause** (`P` key): stop recognizing, e.g. while you talk with your hands.

### Teach (add a sign)

1. Open **Teach**, type the name (e.g. `J` or `OBRIGADO`).
2. Choose **Static** (hold a handshape) or **With movement**.
3. Click **Record** (`Space`). After the 3-2-1 countdown:
   * static: records 1 second,
   * with movement: records from when you start moving until your hands go down,
     or you hold still (max 3 s).
4. **Keep** (`Enter`) or **Discard** (`Backspace`). After Keep, the next take starts by itself.
5. After **5 takes** (10 is better), click **Save**. The sign is recognized
   immediately, and the app asks you to "Now sign `J`".

While recording, the app warns you about **"Hand not visible"**, a static take where
you **moved**, or **"This looks like `I`, are you sure?"**.

To add more takes to an existing sign, type the same name (or use **+ Add takes**
in the Library).

### Library

* All signs with their number of examples. Click a sign to see its examples;
  hover one to replay it. Delete a whole sign or just one bad example.
* Examples marked **⚠ looks like X** are closer to another sign than to their own
  (often a bad take). Deleting them usually improves recognition.
* **Export .zip / Import .zip**: record signs at home, export, then import on the
  demo laptop. Importing **merges** (it never deletes what is already there).

### Status bar and settings

* Top bar: Python connected, FPS, left hand ✔, right hand ✔, face ✔.
* **⚙ Settings**: confidence threshold, camera, mirror view, "accept signs made
  with either hand" (left-handed people can use signs recorded by right-handed
  people), and the height of the rest line.

### Keyboard shortcuts

| Key | Where | Action |
|---|---|---|
| `S` | anywhere | spelling mode on/off |
| `P` | anywhere | pause/resume recognition |
| `Space` | Teach | record a take |
| `Enter` | Teach (review) | keep the take |
| `Backspace` | Teach (review) | discard the take |
| `Backspace` | spelling mode | delete the last letter |
| `Esc` | Teach | cancel |

---

## 3. How it works

```
Browser (index.html)                       Python (FastAPI)
  webcam -> JPEG frames (~15-20 fps) ──ws──> 1. decode frame
                                            2. MediaPipe Holistic -> hands, pose, face
  draws skeleton + results  <──ws── JSON    3. normalize landmarks -> feature vector
                                            4. segmenter: did a sign start / end?
                                            5. k-NN classifier: which sign?
                                            6. teach mode: save examples to disk
```

* **Features** (`app/features.py`): per frame, each hand's shape (20 points
  relative to the wrist, scaled by hand size), each hand's position relative to
  the shoulders (scaled by shoulder width), ~40 face points and the elbows. So
  your position in the frame and your distance to the camera don't matter.
* **Segmenter** (`app/segmenter.py`): a sign **starts** when a hand rises above the
  rest line (or a held hand moves), and **ends** when the hands go down, stay
  still for 0.4 s, or after 3 s. Movements shorter than 0.25 s are ignored. The
  same segmenter cuts the takes in Teach mode, so examples and live attempts match.
* **Classifier** (`app/classifier.py`): k-NN (scikit-learn). Static signs are compared
  by their held handshape; signs with movement by the whole attempt resampled to
  **32 frames**. Distances are scaled by how much your own examples of a sign vary,
  then turned into confidences (with an "unknown" option for "Not recognized").
  Adding a sign = adding examples, so there is no training wait.
* **Storage** (`app/library.py`): `data/signs.json` + one `data/samples/<id>.npz` per
  take. Samples keep the **raw landmarks** (not features), so the feature code can
  improve later without re-recording.

### Project structure

```
app/
  main.py          FastAPI app: page, library API, WebSocket /ws
  session.py       one browser connection: recognize + teach flows
  landmarks.py     MediaPipe wrapper (downloads the model on first run)
  features.py      normalization, resampling to 32 frames
  segmenter.py     start/end of a sign
  classifier.py    k-NN, confidence, "looks like X" check
  library.py       save/load/delete signs, export/import zip
  static/index.html  the whole interface (HTML + CSS + JS)
data/              your signs (signs.json + samples/)
models/            MediaPipe model (downloaded automatically)
tools/import_dataset.py   import videos (e.g. MINDS-Libras) as examples
tests/             automated tests
```

### WebSocket messages

Browser → Python: `frame {jpeg}`, `settings {threshold, mirror, spelling, rest_line, paused}`,
`teach_start {label, kind}`, `teach_record`, `teach_keep`, `teach_discard`,
`teach_save`, `teach_cancel`.

Python → browser: `landmarks {hands, pose, face, up, rest_y, fps, ms}`,
`state {idle|signing|countdown|recording|paused}`,
`prediction {label, confidence, alternatives, recognized}`, `teach {phase, count, ...}`,
`warning {message}`, `library_changed`.

---

## 4. Tests

```bash
source .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest
```

The tests use synthetic landmarks for the logic (features, segmenter, classifier,
library, teach flow), plus the real MediaPipe model on a sample photo for the
WebSocket. The photo is downloaded on the first run; that test is skipped offline.

---

## 5. Importing datasets (after the demo)

Put videos in one folder per sign (`videos/OBRIGADO/*.mp4`, ...) and run:

```bash
python tools/import_dataset.py videos/ --kind dynamic     # or --kind static
python tools/import_dataset.py videos/ --dry-run          # preview only
```

Each video goes through the same MediaPipe + segmenter pipeline as the live camera.

---

## 6. Troubleshooting

| Problem | Fix |
|---|---|
| "Camera not available" | Click the camera icon in Chrome's address bar and allow it. Use `http://localhost:8000`, not the IP address. Close other apps using the camera (Zoom, Teams, ...). |
| "Python disconnected" | The server is not running (start `./run.sh`); the page reconnects by itself. |
| `pip install` fails | Use Python 3.11–3.14. The `run.sh` script picks one automatically. |
| First start fails downloading the model | It needs internet once. Or copy `models/holistic_landmarker.task` from another laptop. |
| Low FPS | Close other heavy apps; plug in the laptop. 12–20 fps is normal. |
| Signs not detected as starting | Raise your hands higher, or lower the rest line in ⚙ Settings. |
| Too many wrong guesses | Raise the confidence threshold, record 10 takes per sign, delete examples marked ⚠. |
| Port 8000 busy | `PORT=8080 ./run.sh` and open `http://localhost:8080`. |
| Want to see MediaPipe's own log lines (hidden by default) | `LIBRAS_VERBOSE=1 ./run.sh` |

Tuning knobs, if needed: `REST_LINE`, `HOLD_TIME`, `STILL_DIST`, `MOVE_DIST` in
`app/segmenter.py`; `UNKNOWN_SCORE` and `K` in `app/classifier.py`.

---

## 7. Demo checklist

**Before the day**

- [ ] Record the base vocabulary with Teach mode: ~10 letters + 5–10 words,
      **10 takes each**, ideally by 2–3 different people.
- [ ] Library → **Export .zip**; on the demo laptop run `./run.sh` once *with internet*,
      then Library → **Import .zip**.
- [ ] Test in the presentation room: light in front of you (not a window behind),
      plain background if possible, shoulders visible in the frame.
- [ ] Practice the demo flow at least twice.
- [ ] Record a screen video of a successful run (backup plan).

**Suggested script (≈ 5 minutes)**

1. Show the skeleton tracking hands and face.
2. Sign 3–4 known words.
3. Fingerspell a short name in spelling mode (`S`).
4. Teach a new sign live: someone from the audience picks a sign, records 5 takes, then signs it.
5. Close with what it can and can't do: it recognizes isolated signs; it is not a Libras translator.
