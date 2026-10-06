from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import os


LOG_DIR = "outputs/2026-09-28/09-28-06/walker_walk_easy/logs"
OUTPUT_DIR = "."


# ---------------------------------------------------------
# Read TensorBoard event file
# ---------------------------------------------------------

event_acc = EventAccumulator(LOG_DIR)
event_acc.Reload()

print("Available scalar tags:")
for tag in event_acc.Tags()["scalars"]:
    print("  ", tag)


# ---------------------------------------------------------
# Extract scalar
# ---------------------------------------------------------

def get_scalar(tag):
    events = event_acc.Scalars(tag)

    steps = [e.step for e in events]
    values = [e.value for e in events]

    return steps, values


# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------

tags = [
    "Episode running reward",
    "Episode reward",
    "Actor loss",
    "Critic loss",
]

for tag in tags:

    steps, values = get_scalar(tag)

    plt.figure(figsize=(10, 6))
    plt.plot(steps, values)

    plt.xlabel("Iteration")
    plt.ylabel(tag)
    plt.title(tag)

    plt.grid(True)
    plt.tight_layout()

    filename = tag.lower().replace(" ", "_") + ".png"
    path = os.path.join(OUTPUT_DIR, filename)

    plt.savefig(path, dpi=200)
    plt.close()

    print(f"Saved: {path}")