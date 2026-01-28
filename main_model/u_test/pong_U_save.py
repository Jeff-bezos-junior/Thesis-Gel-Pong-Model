import turtle
import math
import time
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import os
import inspect

RUN_TIME = 100  # seconds (example)

# ----------------- GLOBAL SPEED CONTROL VARIABLE -----------------
SLOW_FACTOR = 0.5
# -----------------------------------------------------------------

# ----------------- NOISE CONTROL VARIABLE ------------------------
NOISE_FACTOR = 0.2
# -----------------------------------------------------------------

# ---- Folder Saving Control ----
SAVE_ROOT = r"paste_the_path_to_save"
os.makedirs(SAVE_ROOT, exist_ok=True)

# Ask user whether to create a new folder
resp = input("Create a new save folder? (y/n): ").strip().lower()

# Determine latest folder index
existing_folders = [f for f in os.listdir(SAVE_ROOT) if f.startswith("folder")]
next_folder_num = len(existing_folders) + 1

if resp == "y":
    CURRENT_FOLDER = os.path.join(SAVE_ROOT, f"folder{next_folder_num}")
    os.makedirs(CURRENT_FOLDER, exist_ok=True)
    print(f"📁 Created new folder: {CURRENT_FOLDER}")
else:
    # Use the newest folder, or create folder1 if none exist
    if existing_folders:
        latest = sorted(existing_folders)[-1]
        CURRENT_FOLDER = os.path.join(SAVE_ROOT, latest)
        print(f"📂 Using existing folder: {CURRENT_FOLDER}")
    else:
        CURRENT_FOLDER = os.path.join(SAVE_ROOT, "folder1")
        os.makedirs(CURRENT_FOLDER, exist_ok=True)
        print(f"📂 No existing folder found. Created: {CURRENT_FOLDER}")



print("Select Mode:")
print("1: All correct (normal learning)")
print("2: Scrambled paddle")
print("3: Scrambled sensors")
mode = input("Enter choice (1/2/3): ").strip()

if mode == "1":
    MODE = "correct"
elif mode == "2":
    MODE = "scrambled_paddle"
elif mode == "3":
    MODE = "scrambled_sensor"
else:
    MODE = "correct"

win = turtle.Screen()
win.title("FEP Pong Simulation (Parabola Decision Model)")
win.bgcolor("#87CEFA")
win.setup(width=900, height=900)
win.tracer(0)

block_width = 300
block_height = 300

paddle = turtle.Turtle()
paddle.speed(0)
paddle.shape("square")
paddle.color("white")
paddle.shapesize(stretch_wid=(block_height / 20), stretch_len=1)
paddle.penup()
paddle.goto(-290, 0)

ball = turtle.Turtle()
ball.speed(0)
ball.shape("square")
ball.color("white")
ball.penup()
ball.goto(0, 0)

def initialize_ball_speed():
    A = 4
    B = 8
    ball.dx = random.uniform(A / SLOW_FACTOR, B / SLOW_FACTOR)
    ball.dy = random.uniform(1, 8) * random.choice([-1, 1]) / SLOW_FACTOR

initialize_ball_speed()
game_started = True

highlight = turtle.Turtle(); highlight.hideturtle(); highlight.penup()
grid = turtle.Turtle(); grid.hideturtle(); grid.penup(); grid.pensize(3)

current_score = 0
trial_counter = 0
region_hits = {"A": 0, "B": 0, "C": 0}
region_trials = {"A": 0, "B": 0, "C": 0}
region_history = {"A": [], "B": [], "C": []}
region_time = {"A": [], "B": [], "C": []}

# ---------------- MEMORY STATES ----------------
region_mem_value = {"A": None, "B": None, "C": None}
region_last_update = {"A": None, "B": None, "C": None}
region_was_on = {"A": False, "B": False, "C": False}
region_elapsed = {"A": 0.0, "B": 0.0, "C": 0.0}
# ------------------------------------------------

start_time = time.time()

score_display = turtle.Turtle(); score_display.hideturtle(); score_display.penup()
score_display.color("white"); score_display.goto(20, 350)

current_display = turtle.Turtle(); current_display.hideturtle(); current_display.penup()
current_display.color("black"); current_display.goto(350, 280)

# --- Data storage for graph ---
time_data, curr1_data, curr2_data, curr3_data = [], [], [], []

# === NEW === Rally-length logging
rally_lengths = []
rally_times = []
current_rally = 0

# -------------------- FUNCTIONS --------------------
BASELINE = 1.0
BASELINE1 = 10.0
def f_A(t): return BASELINE1
def f_B(t): return BASELINE1
def f_C(t): return BASELINE1
region_funcs = {"A": f_A, "B": f_B, "C": f_C}

def sine_wave_noise(t, sensor_idx):
    s = (0.5 * math.sin(0.15 * t + 1.3 * sensor_idx) +
         0.3 * math.sin(0.90 * t + 0.7 * sensor_idx) +
         0.2 * math.sin(2.40 * t + 0.9 * sensor_idx))
    s += random.uniform(-0.05, 0.05)
    s = max(-1.0, min(1.0, s))
    val = BASELINE + NOISE_FACTOR * s
    return max(BASELINE - NOISE_FACTOR, min(BASELINE + NOISE_FACTOR, val))

def get_ball_region():
    if ball.ycor() > 150: return "A"
    elif ball.ycor() > -150: return "B"
    else: return "C"

def draw_regions():
    grid.clear(); grid.color("black")
    for x in [-300, 0]:
        grid.goto(x, 450); grid.setheading(270)
        grid.pendown(); grid.forward(900); grid.penup()
    for y in [450, 150, -150, -450]:
        grid.goto(-300, y); grid.setheading(0)
        grid.pendown(); grid.forward(600); grid.penup()

def highlight_region():
    row = 0 if ball.ycor() > 150 else 1 if ball.ycor() > -150 else 2
    y = 450 - row * 300
    highlight.clear()
    x_start = -300 if ball.xcor() < 0 else 0
    highlight.goto(x_start, y)
    highlight.fillcolor("#1E90FF")
    highlight.begin_fill()
    for _ in range(2):
        highlight.pendown(); highlight.forward(300); highlight.right(90)
        highlight.forward(300); highlight.right(90); highlight.penup()
    highlight.end_fill(); draw_regions()

def compute_currents(y):
    t_now = time.time() - start_time
    currents = np.array([sine_wave_noise(t_now, i) for i in range(3)])
    active = get_ball_region()
    idx_map = {"A": 0, "B": 1, "C": 2}
    for r in ["A", "B", "C"]:
        idx = idx_map[r]
        is_on = (r == active)
        f_region = region_funcs[r]
        if is_on and region_mem_value[r] is None:
            region_mem_value[r] = f_region(0.0)
            region_last_update[r] = t_now
            region_elapsed[r] = 0.0
            region_was_on[r] = True
        if is_on:
            if not region_was_on[r]:
                region_last_update[r] = t_now
                region_was_on[r] = True
            dt = t_now - (region_last_update[r] or t_now)
            region_elapsed[r] += dt
            region_last_update[r] = t_now
            val = f_region(region_elapsed[r])
            region_mem_value[r] = val
            currents[idx] = val
        else:
            region_was_on[r] = False
    return currents

last_paddle_update = time.time()
paddle_update_dt = 0.1

def decide_paddle_y():
    y = ball.ycor()
    currents = compute_currents(y)
    t_now = time.time() - start_time
    time_data.append(t_now)
    curr1_data.append(currents[0])
    curr2_data.append(currents[1])
    curr3_data.append(currents[2])
    lowRangeC, upRangeC = -7.0, 7.6
    norm_currents = np.clip((currents - lowRangeC)/(upRangeC - lowRangeC), 0, 1)
    x_positions = np.array([-1.0, 0.0, 1.0])
    a,b,c = np.polyfit(x_positions, norm_currents, 2)
    x_samples = np.linspace(-1,1,200)
    y_samples = a*x_samples**2 + b*x_samples + c
    x_vertex = x_samples[np.argmax(y_samples)]
    paddle_y = -np.clip(x_vertex,-1,1)*300
    return paddle_y, currents

def move_paddle_instant():
    global last_paddle_update
    now = time.time()
    target, currents = decide_paddle_y()
    if now - last_paddle_update >= paddle_update_dt:
        last_paddle_update = now
        if MODE == "scrambled_paddle":
            target = random.choice([-300,0,300])
        paddle.sety(max(-300,min(300,target)))
    return target, currents

def normalize_velocity():
    BASE_MIN,BASE_MAX,BASE_TARGET = 2.0,12.0,6.0
    MIN_SPEED=max(0.01,BASE_MIN/SLOW_FACTOR)
    MAX_SPEED=BASE_MAX/SLOW_FACTOR
    DESIRED_SPEED=BASE_TARGET/SLOW_FACTOR
    speed=math.sqrt(ball.dx**2+ball.dy**2)
    if speed<MIN_SPEED or speed>MAX_SPEED:
        scale=DESIRED_SPEED/speed
        ball.dx*=scale; ball.dy*=scale

def update_score():
    score_display.clear()
    score_display.write(f"{current_score}",align="center",font=("Arial",16,"bold"))

def update_current_display(currents,paddle_y):
    current_display.clear()
    elapsed = time.time() - start_time
    text=(f"Currents:\nTop: {currents[0]:.3f} mA\n"
          f"Middle: {currents[1]:.3f} mA\n"
          f"Bottom: {currents[2]:.3f} mA\n"
          f"Paddle Y: {paddle_y:.1f}\n"
          f"Time: {elapsed:6.1f} s")
    current_display.write(text,align="center",font=("Courier",12,"bold"))

# === NEW === Rally length plot
def plot_rally_length():
    if len(rally_lengths) < 4:
        print("Not enough rally data.")
        return
    df = pd.DataFrame({"time": rally_times, "rally_length": rally_lengths})
    plt.figure(figsize=(10,5))
    plt.plot(df["time"], df["rally_length"], color="orange", alpha=0.7, label="Rally Length")
    plt.xlabel("Time (s)")
    plt.ylabel("Rally Length")
    plt.title("Rally Length Over Time (Global)")
    plt.grid(True)

    mid = len(df)//2
    pre = df["rally_length"][:mid]
    post = df["rally_length"][mid:]
    stat, p = mannwhitneyu(pre, post, alternative="less")
    plt.text(0.02, 0.92, f"p = {p:.7f}", transform=plt.gca().transAxes,
             fontsize=12, fontweight="bold",
             bbox=dict(facecolor="white", edgecolor="black", alpha=0.7))
    plt.legend()
    plt.show()


import os

def save_rally_data():
    """Save rally length data into CURRENT_FOLDER as verN.csv, even if empty."""
    os.makedirs(CURRENT_FOLDER, exist_ok=True)

    # Count existing version files
    existing = [f for f in os.listdir(CURRENT_FOLDER)
                if f.startswith("ver") and f.endswith(".csv")]
    next_ver = len(existing) + 1


    save_path = os.path.join(CURRENT_FOLDER, f"ver{next_ver}.csv")

    # Always save, even if no rallies happened
    df = pd.DataFrame({
        "time": rally_times if rally_times else [0],
        "rally_length": rally_lengths if rally_lengths else [0]
    })

    df.to_csv(save_path, index=False)
    print(f"✅ Rally data saved to: {save_path}")



def plot_hit_rate():
    fig, axs = plt.subplots(3, 1, figsize=(8, 10))
    total_elapsed = time.time() - start_time
    for i, r in enumerate(["A", "B", "C"]):
        hist, times = region_history[r], region_time[r]
        if not hist: continue
        elapsed_times = np.array(times)
        if len(elapsed_times) > 0:
            elapsed_times = elapsed_times - elapsed_times[0]
            if elapsed_times[-1] > 0:
                elapsed_times = elapsed_times / elapsed_times[-1] * total_elapsed
        axs[i].plot(elapsed_times, hist, marker="o", linewidth=2)
        axs[i].set_title(f"Region {r} Hit Rate vs Time")
        axs[i].set_xlabel("Elapsed Time (s)")
        axs[i].set_ylabel("Hit Rate")
        axs[i].set_ylim(0, 1.05)
        axs[i].grid(True)
    plt.tight_layout()
    plt.show()

def plot_currents_after_run():
    df = pd.DataFrame({"time": time_data,
                       "current1": curr1_data,
                       "current2": curr2_data,
                       "current3": curr3_data})
    df.to_csv("currents_log.csv", index=False)
    print("✅ Current data saved to 'currents_log.csv'")
    plt.figure(figsize=(10,6))
    plt.plot(df["time"], df["current1"], label="Region A (Top)")
    plt.plot(df["time"], df["current2"], label="Region B (Middle)")
    plt.plot(df["time"], df["current3"], label="Region C (Bottom)")
    plt.xlabel("Time (s)")
    plt.ylabel("Current (mA)")
    plt.title("Independent Region Currents (Function-driven)")
    plt.legend(); plt.grid(True)
    plt.show()

def quit_game():
    win.bye()
    save_rally_data()
    plot_currents_after_run()
    plot_hit_rate()
    plot_rally_length()  # === NEW ===

win.listen(); win.onkeypress(quit_game, "q")
draw_regions(); update_score(); update_current_display(np.array([0,0,0]),0)

# ---------------- MAIN LOOP ----------------
while True:
    win.update()
    if game_started:
        ball.setx(ball.xcor()+ball.dx)
        ball.sety(ball.ycor()+ball.dy)
        highlight_region()
        paddle_y,currents=move_paddle_instant()
        update_current_display(currents,paddle_y)

        if ball.ycor()>435: ball.sety(435); ball.dy*=-1; normalize_velocity()
        if ball.ycor()<-435: ball.sety(-435); ball.dy*=-1; normalize_velocity()
        if ball.xcor()>285: ball.setx(285); ball.dx*=-1; normalize_velocity()

        region=get_ball_region()
        if -290<ball.xcor()<-260 and (paddle.ycor()-150)<ball.ycor()<(paddle.ycor()+150):
            ball.setx(-260); ball.dx*=-1; normalize_velocity()
            current_score+=1; region_hits[region]+=1; region_trials[region]+=1; update_score()
            current_rally += 1  # === NEW === count consecutive hits

        if ball.xcor()<-290:
            region_trials[region]+=1; trial_counter+=1
            # === NEW === store rally length each time ball missed
            rally_lengths.append(current_rally)
            rally_times.append(time.time()-start_time)
            current_rally = 0
            # -----------------------------
            current_t=time.time()-start_time
            for r in ["A","B","C"]:
                if region_trials[r]>0:
                    rate=region_hits[r]/region_trials[r]
                    region_history[r].append(rate)
                    region_time[r].append(time.time() - start_time)
            current_score=0; ball.goto(0,0); initialize_ball_speed(); update_score()
        normalize_velocity()

        if time.time() - start_time >= RUN_TIME:
            quit_game()
            break

