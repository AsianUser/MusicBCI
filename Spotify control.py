"""
EEG Spotify Controller
======================
Maps 8 integer outputs from your EEG face-muscle ML model to Spotify controls on Mac.
All actions use AppleScript — no window focus switching, terminal stays active.

Usage:
    from spotify_eeg_controller import handle_action

    handle_action(1)  # Play / Pause
    handle_action(2)  # Volume Up
    handle_action(3)  # Volume Down
    handle_action(4)  # Skip (next track)
    handle_action(5)  # Go Back (previous track)
    handle_action(6)  # Shuffle Toggle
    handle_action(7)  # Mute Toggle
    handle_action(8)  # Repeat Toggle

Requirements:
    No pip installs needed — uses only subprocess (built-in) and osascript (built-in on Mac)
"""

import subprocess


# --- Config ---
VOLUME_STEP = 10  # % per volume up/down call


def _run_applescript(script: str) -> str:
    """Run an AppleScript and return stdout."""
    result = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True, text=True
    )
    return result.stdout.strip()


# --- Individual action functions ---

def play_pause():
    """Action 1: Toggle Play / Pause."""
    print("[EEG] Action 1 → Play / Pause")
    _run_applescript('tell application "Spotify" to playpause')


def volume_up():
    """Action 2: Volume Up (fixed step)."""
    print(f"[EEG] Action 2 → Volume Up (+{VOLUME_STEP}%)")
    current = int(_run_applescript('tell application "Spotify" to return sound volume'))
    new_vol = min(100, current + VOLUME_STEP)
    _run_applescript(f'tell application "Spotify" to set sound volume to {new_vol}')
    print(f"  Volume: {current}% → {new_vol}%")


def volume_down():
    """Action 3: Volume Down (fixed step)."""
    print(f"[EEG] Action 3 → Volume Down (-{VOLUME_STEP}%)")
    current = int(_run_applescript('tell application "Spotify" to return sound volume'))
    new_vol = max(0, current - VOLUME_STEP)
    _run_applescript(f'tell application "Spotify" to set sound volume to {new_vol}')
    print(f"  Volume: {current}% → {new_vol}%")


def skip_track():
    """Action 4: Skip to next track."""
    print("[EEG] Action 4 → Skip (Next Track)")
    _run_applescript('tell application "Spotify" to next track')


def previous_track():
    """Action 5: Go back to previous track."""
    print("[EEG] Action 5 → Previous Track")
    _run_applescript('tell application "Spotify" to previous track')


def shuffle_toggle():
    """Action 6: Toggle Shuffle on/off."""
    print("[EEG] Action 6 → Shuffle Toggle")
    current = _run_applescript('tell application "Spotify" to return shuffling')
    new_state = "false" if current == "true" else "true"
    _run_applescript(f'tell application "Spotify" to set shuffling to {new_state}')
    print(f"  Shuffle: {'ON' if new_state == 'true' else 'OFF'}")


def mute_toggle():
    """Action 7: Toggle Mute (saves and restores volume)."""
    print("[EEG] Action 7 → Mute Toggle")
    current = int(_run_applescript('tell application "Spotify" to return sound volume'))

    if current == 0:
        restore_vol = getattr(mute_toggle, "_last_volume", 50)
        _run_applescript(f'tell application "Spotify" to set sound volume to {restore_vol}')
        print(f"  Unmuted → restored to {restore_vol}%")
    else:
        mute_toggle._last_volume = current
        _run_applescript('tell application "Spotify" to set sound volume to 0')
        print(f"  Muted (was {current}%)")


def repeat_toggle():
    """Action 8: Cycle Repeat mode (off → on → off)."""
    print("[EEG] Action 8 → Repeat Toggle")
    current = _run_applescript('tell application "Spotify" to return repeating')
    new_state = "false" if current == "true" else "true"
    _run_applescript(f'tell application "Spotify" to set repeating to {new_state}')
    print(f"  Repeat: {'ON' if new_state == 'true' else 'OFF'}")


# --- Dispatch map ---

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
    """
    Main entry point. Pass in the integer output from your EEG model (1–8).

    Args:
        action_id: Integer from 1 to 8 corresponding to a Spotify action.
    """
    if action_id not in ACTIONS:
        print(f"[EEG] Warning: action_id {action_id} is out of range (expected 1–8). Ignoring.")
        return

    ACTIONS[action_id]()


# --- Interactive Test Mode ---

if __name__ == "__main__":
    print("=" * 40)
    print("  EEG Spotify Controller - Test Mode")
    print("=" * 40)
    print("  1  →  Play / Pause")
    print("  2  →  Volume Up")
    print("  3  →  Volume Down")
    print("  4  →  Skip (Next Track)")
    print("  5  →  Previous Track")
    print("  6  →  Shuffle Toggle")
    print("  7  →  Mute Toggle")
    print("  8  →  Repeat Toggle")
    print("  q  →  Quit")
    print("=" * 40)
    print("Make sure Spotify is open!\n")

    while True:
        raw = input("Enter action (1-8) or 'q' to quit: ").strip()

        if raw.lower() == "q":
            print("Exiting test mode.")
            break

        if not raw.isdigit():
            print("  ⚠ Please enter a number between 1 and 8.\n")
            continue

        handle_action(int(raw))
        print()