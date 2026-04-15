import random
from psychopy import data, visual, core, event, logging
import serial

# set up link to Wearable Headset
PORT = "COM8"
try:
    mmbts = serial.Serial(PORT, baudrate=115200, timeout=1)
    print("connected to trigger")
except Exception as e:
    mmbtts = None
    print("fail connect trigger")


triggers = [
    "blink",
    "winkLeft",
    "winkRight",
    "jawFull",
    "jawLeft",
    "jawRight",
    "faceLeft",
    "faceRight",
]
triggers_int = [1, 2, 3, 4, 5, 6, 7, 8]
event_length = 2  # how long trigger on screen (seconds)

triggers_visual = [
    "Blink!",
    "Wink Left!",
    "Wink Right!",
    "Clench Whole Jaw! (HOLD)",
    "Clench Left Jaw! (HOLD)",
    "Clench Right Jaw! (HOLD)",
    "Scrunch Left Face! (HOLD)",
    "Scrunch Right Face! (HOLD)",
]
# what the subject reads

avg_delay = 2  # seconds between triggers
spread_delay = 1  # how much variance (seconds) between delays
min_delay = avg_delay - spread_delay
max_delay = avg_delay + spread_delay

prime_time = 2  # how long primer appears before trigger (seconds)

blink_interval = 4  # blink will appear at least once every 4 triggers

experiment_length = 210  # avg length of experiment (seconds)
triggers_per_trial = 30  # how much before guarantee all triggers seen again

total_trials = int(experiment_length / triggers_per_trial)

# set up window
win = visual.Window(
    fullscr=False, color=[255, 255, 255], units="height"
)  # set fullscr=True for real experiment
text = visual.TextStim(win, text="", height=0.12, color="black")
# experiment clock
global_clock = core.Clock()

# allow ESC to cancel experiment anywhere
event.globalKeys.add(key="escape", func=core.quit)

inst = visual.TextStim(
    win,
    text="You will see text on the screen.\nRead the instructions on the Primer Screen.\nPerform the indicated facial action when the prompt appears.\n If you see '''(HOLD)''' at any point, hold that facial action for the duration of the text\nPress ESC to quit anytime.",
    height=0.05,
    color="black",
)
inst.draw()
win.flip()
keys = event.waitKeys(keyList=["space", "escape"])
if "space" in keys:
    win.flip()
if "escape" in keys:
    win.close()
    core.quit()

# ------------------
# Main Trials Loop
# ------------------
for trial_count in range(0, total_trials):

    # creates the triggers & order to be seen per trial
    # Separate blink from other triggers
    blink_trigger = "blink"
    other_triggers = [t for t in triggers if t != blink_trigger]

    trial_trigger_list = []

    # Number of blocks
    num_blocks = triggers_per_trial // blink_interval

    for _ in range(num_blocks):
        block = [blink_trigger]  # guarantee one blink

        # fill remaining positions in block randomly (excluding blink)
        for _ in range(blink_interval - 1):
            block.append(random.choice(other_triggers))

        random.shuffle(block)
        trial_trigger_list.extend(block)

    # If triggers_per_trial is not perfectly divisible
    remaining = triggers_per_trial - len(trial_trigger_list)
    if remaining > 0:
        extras = [random.choice(triggers) for _ in range(remaining)]
        trial_trigger_list.extend(extras)

    random.shuffle(trial_trigger_list)

    print(trial_trigger_list)

    for trigger in trial_trigger_list:
        j = triggers.index(trigger)
        visual_label = triggers_visual[j]
        prime_label = f"Get ready for:\n {visual_label}"

        delay = random.uniform(min_delay, max_delay)
        tstart_wait = global_clock.getTime()
        while global_clock.getTime() - tstart_wait < delay:
            # small sleep so we can respond to ESC quickly via globalKeys
            core.wait(0.01)

        # prime the user
        text.text = prime_label
        text.draw()
        win.flip()

        tstart_wait = global_clock.getTime()
        while global_clock.getTime() - tstart_wait < prime_time:
            # prime screen for user
            core.wait(0.01)

        # reset screen
        win.flip()

        # prep the marker in the eeg data
        win.callOnFlip(mmbts.write, bytes([j]))

        # prepare stim & reveal
        text.text = visual_label
        text.draw()
        win.flip()

        # show stim for time
        # collect any keypress during stimulus (any key accepted)
        resp_key = None
        resp_rt = None
        stim_onset = global_clock.getTime()
        while global_clock.getTime() - stim_onset < event_length:
            keys = event.getKeys(timeStamped=global_clock)
            if keys:
                # take first key only
                keyname, keytime = keys[0]
                resp_key = keyname
                resp_rt = keytime - stim_onset
                break
            core.wait(0.005)

        # clear screen
        win.flip()


print("\n")
# close serial if open
if mmbtts is not None:
    try:
        mmbtts.close()
    except Exception:
        pass


# goodbye screen
thank = visual.TextStim(
    win, text="Thanks! End of experiment.\nPress any key to quit.", height=0.05
)
thank.draw()
win.flip()
while global_clock.getTime() - tstart_wait < 5:
    # small sleep so we can respond to ESC quickly via globalKeys
    core.wait(0.01)
win.close()
core.quit()
print(f"fin")
