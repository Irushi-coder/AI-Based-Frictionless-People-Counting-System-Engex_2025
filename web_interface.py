from flask import Flask, jsonify, render_template
from flask_cors import CORS
from threading import Lock
from counters_shared import zone_counters, counter_lock
import os
import json
import atexit
import time
from threading import Thread

# Persistence settings
PERSIST_PATH = os.path.join(os.path.dirname(__file__), 'counters.json')
SAVE_INTERVAL_SECONDS = 5
_autosave_started = False

def load_counters_from_file():
    """Load saved counters and merge into zone_counters."""
    if not os.path.exists(PERSIST_PATH):
        return
    try:
        with open(PERSIST_PATH, 'r') as f:
            data = json.load(f)
        if isinstance(data, dict):
            with counter_lock:
                # Merge without dropping existing keys that may be initialized elsewhere
                for k, v in data.items():
                    zone_counters[k] = v
    except Exception as e:
        print(f"[warn] Failed to load {PERSIST_PATH}: {e}")

def save_counters_to_file():
    """Atomically save counters to disk."""
    try:
        with counter_lock:
            data = dict(zone_counters)
        tmp = PERSIST_PATH + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(data, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, PERSIST_PATH)
    except Exception as e:
        print(f"[warn] Failed to save {PERSIST_PATH}: {e}")

def _autosave_loop():
    while True:
        time.sleep(SAVE_INTERVAL_SECONDS)
        save_counters_to_file()

def _ensure_autosave_started():
    global _autosave_started
    if _autosave_started:
        return
    t = Thread(target=_autosave_loop, daemon=True)
    t.start()
    _autosave_started = True

# Load persisted counters on startup and set up autosave and shutdown save
load_counters_from_file()
_ensure_autosave_started()
atexit.register(save_counters_to_file)

detection_booted = False

def ensure_detection():
    global detection_booted
    if detection_booted:
        return
    from Multiple_zones import start_detection  # import inside to avoid early heavy init
    start_detection()
    detection_booted = True

app = Flask(__name__, template_folder='templates')  # Specify the templates folder
CORS(app)  # Enable CORS for frontend-backend communication

@app.route('/')
def index():
    ensure_detection()
    return render_template('index.html')  # Render the index.html file

@app.route('/api/zone_counters')
def get_zone_counters():
    with counter_lock:
        return jsonify(zone_counters)

if __name__ == '__main__':
    # IMPORTANT: disable reloader to prevent double start of detection threads
    app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)


    