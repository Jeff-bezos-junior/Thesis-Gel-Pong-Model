import turtle
import math
import time
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ----------------- GLOBAL SPEED CONTROL VARIABLE -----------------
SLOW_FACTOR = 2
# -----------------------------------------------------------------

# ----------------- NOISE CONTROL VARIABLE ------------------------
NOISE_FACTOR = 0.5
# -----------------------------------------------------------------

# ----------------- MEMORY / SPEED RETENTION ----------------------
# 1.0 -> perfect current retention, fastest paddle
# 0.5 -> each new ON episode is half amplitude, medium paddle speed
# 0.0 -> no memory, very slow paddle
RETENTION_FACTOR = 0.9
# -----------------------------------------------------------------

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

# ----------------- PADDLE & BALL -----------------
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

highlight = turtle.Turtle()
highlight.hideturtle()
highlight.penup()

grid = turtle.Turtle()
grid.hideturtle()
grid.penup()
grid.pensize(3)

current_score = 0
trial_counter = 0
region_hits = {"A": 0, "B": 0, "C": 0}
region_trials = {"A": 0, "B": 0, "C": 0}
region_history = {"A": [], "B": [], "C": []}
region_time = {"A": [], "B": [], "C": []}

# ============ MEMORY STATE ============

# Last ON value for plotting/debug (not strictly necessary)
region_mem_value = {"A": None, "B": None, "C": None}

# Whether region was ON in previous step (for detecting ON edge)
region_was_on = {"A": False, "B": False, "C": False}

# Time when the region first EVER turned ON
# (we keep this fixed so exponential is continuous over the run)
region_first_on_time = {"A": None, "B": None, "C": None}

# Multiplicative memory scale per region (decays with RETENTION_FACTOR
# on each new ON-episode, but stays 1.0 when RETENTION_FACTOR=1)
region_mem_scale = {"A": 1.0, "B": 1.0, "C": 1.0}

# ======================================

start_time = time.time()

score_display = turtle.Turtle()
score_display.hideturtle()
score_display.penup()
score_display.color("white")
score_display.goto(20, 350)

current_display = turtle.Turtle()
current_display.hideturtle()
current_display.penup()
current_display.color("black")
current_display.goto(350, 280)

# logging arrays
time_data, curr1_data, curr2_data, curr3_data = [], [], [], []

# ---------- Exponential decay functions ----------
BASELINE = 0.32
AMPLITUDE = 7.32
TAU = 123.47
MAX_SINE_AMPLITUDE = 0.35

def f_A(t): return AMPLITUDE * math.exp(-t / TAU) + BASELINE
def f_B(t): return AMPLITUDE * math.exp(-t / TAU) + BASELINE
def f_C(t): return AMPLITUDE * math.exp(-t / TAU) + BASELINE

region_funcs = {"A": f_A, "B": f_B, "C": f_C}

# sine noise function
def sine_wave_noise(t, sensor_idx):
    amp_1 = 0.12 * math.sin(0.1 * t + sensor_idx * 1.5)
    amp_2 = 0.15 * math.sin(0.8 * t + sensor_idx * 0.5)
    amp_3 = 0.08 * math.sin(2.5 * t + sensor_idx * 0.9)
    sine_sum = amp_1 + amp_2 + amp_3
    normalized = sine_sum / MAX_SINE_AMPLITUDE
    rand_comp = random.uniform(-0.05, 0.05) / MAX_SINE_AMPLITUDE
    return (normalized + rand_comp) * NOISE_FACTOR

def get_ball_region():
    if ball.ycor() > 150:
        return "A"
    elif ball.ycor() > -150:
        return "B"
    return "C"

def draw_regions():
    grid.clear()
    grid.color("black")

    for x in [-300, 0]:
        grid.goto(x, 450)
        grid.setheading(270)
        grid.pendown()
        grid.forward(900)
        grid.penup()

    for y in [450, 150, -150, -450]:
        grid.goto(-300, y)
        grid.setheading(0)
        grid.pendown()
        grid.forward(600)
        grid.penup()

def highlight_region():
    row = 0 if ball.ycor() > 150 else 1 if ball.ycor() > -150 else 2
    y = 450 - row * 300
    highlight.clear()
    x_start = -300 if ball.xcor() < 0 else 0
    highlight.goto(x_start, y)
    highlight.fillcolor("#1E90FF")
    highlight.begin_fill()
    for _ in range(2):
        highlight.pendown()
        highlight.forward(300)
        highlight.right(90)
        highlight.forward(300)
        highlight.right(90)
        highlight.penup()
    highlight.end_fill()
    draw_regions()

# ============ CURRENT COMPUTATION =============

def compute_currents():
    """
    - Each region has a continuous exponential f(t) from its first-ever ON time.
    - When RETENTION_FACTOR == 1.0, region_mem_scale stays 1.0, so each
      ON episode continues the same exponential envelope => same plateau
      heights between consecutive ON episodes (like your graph), with
      very slow long-term decay.
    - When RETENTION_FACTOR < 1.0, mem_scale is multiplied by that factor
      at each new ON edge in that region, so pulses get shorter/weaker.
    """
    t_now = time.time() - start_time

    # OFF-state currents: baseline + noise
    currents = np.array([
        sine_wave_noise(t_now, 0) + BASELINE,
        sine_wave_noise(t_now, 1) + BASELINE,
        sine_wave_noise(t_now, 2) + BASELINE
    ])

    active = get_ball_region()
    idx_map = {"A": 0, "B": 1, "C": 2}

    for r in ["A", "B", "C"]:
        idx = idx_map[r]
        is_on = (r == active)
        f_region = region_funcs[r]

        if is_on:
            # First time this region was EVER activated -> set time origin
            if region_first_on_time[r] is None:
                region_first_on_time[r] = t_now

            # Detect ON edge (OFF -> ON)
            if not region_was_on[r]:
                region_was_on[r] = True
                # Apply memory decay on new ON episode (except first)
                if region_first_on_time[r] is not None:
                    region_mem_scale[r] *= RETENTION_FACTOR

            # Continuous time since first ON for this region
            t_mem = t_now - region_first_on_time[r]
            base_val = f_region(t_mem)

            scale = region_mem_scale[r]
            # Final current: baseline + scaled exponential above baseline
            val = BASELINE + scale * (base_val - BASELINE)

            region_mem_value[r] = val
            currents[idx] = val
        else:
            # Region OFF
            region_was_on[r] = False

    return currents

# ===================================================

last_paddle_update = time.time()
paddle_update_dt = 0.1

# ⭐⭐⭐ FIXED-SPEED RETENTION-BASED PADDLE CONTROL ⭐⭐⭐
def move_paddle_retention(active_region):
    """
    Paddle speed is a fixed function of RETENTION_FACTOR:
    - RETENTION_FACTOR = 1.0 -> instant jump to target (within dt)
    - RETENTION_FACTOR = 0.5 -> 50% of gap per update
    - RETENTION_FACTOR near 0 -> very slow
    """
    global last_paddle_update

    now = time.time()
    if now - last_paddle_update < paddle_update_dt:
        return paddle.ycor()
    last_paddle_update = now

    # Target tracking
    if MODE == "scrambled_paddle":
        target_y = random.choice([-300, 0, 300])
    else:
        target_y = ball.ycor()

    # Speed fraction directly from RETENTION_FACTOR, clamped [0,1]
    speed_frac = max(0.0, min(1.0, RETENTION_FACTOR))

    cur_y = paddle.ycor()
    new_y = cur_y + speed_frac * (target_y - cur_y)

    # Clamp paddle within arena
    new_y = max(-300, min(300, new_y))
    paddle.sety(new_y)

    return new_y
# =======================================================

def normalize_velocity():
    BASE_MIN, BASE_MAX, BASE_TARGET = 2.0, 12.0, 6.0
    MIN_SPEED = max(0.01, BASE_MIN / SLOW_FACTOR)
    MAX_SPEED = BASE_MAX / SLOW_FACTOR
    DESIRED_SPEED = BASE_TARGET / SLOW_FACTOR
    speed = math.sqrt(ball.dx**2 + ball.dy**2)
    if speed < MIN_SPEED or speed > MAX_SPEED:
        scale = DESIRED_SPEED / speed
        ball.dx *= scale
        ball.dy *= scale

def update_score():
    score_display.clear()
    score_display.write(f"{current_score}", align="center", font=("Arial", 16, "bold"))

def update_current_display(currents, paddle_y):
    current_display.clear()
    text = (f"Currents:\n"
            f"Top: {currents[0]:.3f} mA\n"
            f"Middle: {currents[1]:.3f} mA\n"
            f"Bottom: {currents[2]:.3f} mA\n"
            f"Paddle Y: {paddle_y:.1f}")
    current_display.write(text, align="center", font=("Courier", 12, "bold"))

# --- Logging and plot functions ---

def plot_hit_rate():
    fig, axs = plt.subplots(3, 1, figsize=(8, 10))
    for i, r in enumerate(["A", "B", "C"]):
        hist, times = region_history[r], region_time[r]
        if hist:
            axs[i].plot(times, hist, marker="o", linewidth=2)
        axs[i].set_title(f"Region {r} Hit Rate vs Time")
        axs[i].set_xlabel("Time (s)")
        axs[i].set_ylabel("Hit Rate")
        axs[i].set_ylim(0, 1.05)
        axs[i].grid(True)
    plt.tight_layout()
    plt.show()

def plot_currents_after_run():
    df = pd.DataFrame({
        "time": time_data,
        "current1": curr1_data,
        "current2": curr2_data,
        "current3": curr3_data
    })
    df.to_csv("currents_log.csv", index=False)
    print("Saved to currents_log.csv")

    plt.figure(figsize=(10, 6))
    plt.plot(df["time"], df["current1"], label="Region A")
    plt.plot(df["time"], df["current2"], label="Region B")
    plt.plot(df["time"], df["current3"], label="Region C")
    plt.xlabel("Time (s)")
    plt.ylabel("Current (mA)")
    plt.title("Independent Region Currents")
    plt.legend()
    plt.grid(True)
    plt.show()

def quit_game():
    win.bye()
    plot_currents_after_run()
    plot_hit_rate()

win.listen()
win.onkeypress(quit_game, "q")

draw_regions()
update_score()
update_current_display(np.array([0, 0, 0]), 0)

# ====================== MAIN LOOP ======================
while True:
    win.update()

    if game_started:
        # Move ball
        ball.setx(ball.xcor() + ball.dx)
        ball.sety(ball.ycor() + ball.dy)

        highlight_region()

        # Currents
        currents = compute_currents()
        t_now = time.time() - start_time
        time_data.append(t_now)
        curr1_data.append(currents[0])
        curr2_data.append(currents[1])
        curr3_data.append(currents[2])

        region = get_ball_region()

        # Paddle
        paddle_y = move_paddle_retention(region)

        update_current_display(currents, paddle_y)

        # Wall collisions
        if ball.ycor() > 435:
            ball.sety(435)
            ball.dy *= -1
            normalize_velocity()

        if ball.ycor() < -435:
            ball.sety(-435)
            ball.dy *= -1
            normalize_velocity()

        if ball.xcor() > 285:
            ball.setx(285)
            ball.dx *= -1
            normalize_velocity()

        # Paddle collision
        region = get_ball_region()
        if -290 < ball.xcor() < -260 and (paddle.ycor() - 150) < ball.ycor() < (paddle.ycor() + 150):
            ball.setx(-260)
            ball.dx *= -1
            normalize_velocity()

            current_score += 1
            region_hits[region] += 1
            region_trials[region] += 1
            update_score()

        # Miss
        if ball.xcor() < -290:
            region_trials[region] += 1
            current_t = time.time() - start_time

            for r in ["A", "B", "C"]:
                if region_trials[r] > 0:
                    rate = region_hits[r] / region_trials[r]
                    region_history[r].append(rate)
                    region_time[r].append(current_t)

            current_score = 0
            ball.goto(0, 0)
            initialize_ball_speed()
            update_score()

        normalize_velocity()
