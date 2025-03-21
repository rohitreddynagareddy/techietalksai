import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Define task durations
tasks = [
    ("SEQUENTIAL START", 0, 0),
    ("greet() starts", 1, 0),
    ("greet() waits 5s", 6, 0),
    ("greet() ends", 6, 0),
    ("count() starts", 6, 1),
    ("count() step 1", 7, 1),
    ("count() step 2", 8, 1),
    ("count() step 3", 9, 1),
    ("count() step 4", 10, 1),
    ("count() step 5", 11, 1),
    ("count() step 6", 12, 1),
    ("count() step 7", 13, 1),
    ("count() step 8", 14, 1),
    ("count() ends", 14, 1),
    ("CONCURRENT START", 14, 0),
    ("greet() starts", 15, 2),
    ("count() starts", 15, 3),
    ("count() step 1", 16, 3),
    ("count() step 2", 17, 3),
    ("count() step 3", 18, 3),
    ("count() step 4", 19, 3),
    ("count() step 5", 20, 3),
    ("greet() ends", 20, 2),
    ("count() step 6", 21, 3),
    ("count() step 7", 22, 3),
    ("count() step 8", 23, 3),
    ("count() ends", 23, 3),
]

# Plot setup
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(0, 24)
ax.set_ylim(-1, 5)
ax.set_xlabel("Time (seconds)")
ax.set_ylabel("Execution Flow")
ax.set_title("Execution Flow of Sequential and Concurrent Async Functions")
ax.grid(True, linestyle="--", alpha=0.6)

# Colors for different tasks
colors = {
    "greet()": "royalblue",
    "count()": "tomato",
    "CONCURRENT START": "green",
    "SEQUENTIAL START": "purple",
}

# Plot task executions
for task, start_time, y_pos in tasks:
    color = colors.get(task.split()[0], "gray")
    ax.add_patch(patches.Rectangle((start_time, y_pos), 1, 0.8, color=color, alpha=0.7))
    ax.text(start_time + 0.2, y_pos + 0.3, task, fontsize=8, color="white")

# Save as SVG
plt.savefig("/mnt/data/execution_flow.svg", format="svg")
plt.show()
