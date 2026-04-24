import time
from pathlib import Path

# import spotipy
# from spotipy.oauth2 import SpotifyOAuth
import keyboard

# --- Config ---

SHARED_FILE = Path("tmp/eeg_action.txt")

VOLUME_STEP = 10


# --- Actions ---


def play_pause():
    print("[EEG] Action 1 → Play / Pause")
    keyboard.send("play/pause media")


def volume_up():
    print("[EEG] Action 2 → Volume Up")
    for _ in range(VOLUME_STEP):
        keyboard.send("volume up")


def volume_down():
    print("[EEG] Action 3 → Volume Down")
    for _ in range(VOLUME_STEP):
        keyboard.send("volume down")


def skip_track():
    print("[EEG] Action 4 → Next Track")
    keyboard.send("next track")


def previous_track():
    print("[EEG] Action 5 → Previous Track")
    keyboard.send("previous track")


def shuffle_toggle():
    print("[EEG] Action 6 → Shuffle (Not supported via media keys)")
    # Optional: map to something else if you want
    # Example: keyboard.send("s")


def mute_toggle():
    print("[EEG] Action 7 → Mute Toggle")
    keyboard.send("volume mute")


def repeat_toggle():
    print("[EEG] Action 8 → Repeat (Not supported via media keys)")
    # Optional: map to something else


# --- Dispatch ---

ACTIONS = {
    1: play_pause,
    2: volume_up,
    3: volume_down,
    4: skip_track,
    5: previous_track,
    6: shuffle_toggle,
    7: mute_toggle,
    8: repeat_toggle,
}


def handle_action(action_id: int):
    if action_id not in ACTIONS:
        print(f"Invalid action: {action_id}")
        return

    ACTIONS[action_id]()


# --- Listener loop ---
POLL_INTERVAL = 1.5  # was 0.05

# --- Listener loop ---

if __name__ == "__main__":
    print("── EEG Media Controller (Windows) ──")
    print(f"Polling {SHARED_FILE} every {int(POLL_INTERVAL * 1000)} ms\n")

    while not SHARED_FILE.exists():
        print("Waiting for eeg_live.py to start…")
        time.sleep(1)

    # last_seen = "0"  # OLD: removed

    last_trigger_time = 0
    COOLDOWN = 0.3  # seconds

    while True:
        raw = SHARED_FILE.read_text().strip()
        now = time.time()

        # OLD:
        # if raw != "0" and raw != last_seen:

        # NEW:
        if raw != "0" and (now - last_trigger_time > COOLDOWN):
            print(f"\nReceived action: {raw}")
            handle_action(int(raw))
            last_trigger_time = now

        # last_seen = raw  # OLD: removed
        time.sleep(POLL_INTERVAL)
