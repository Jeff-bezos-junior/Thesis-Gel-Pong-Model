"""
Main Model 

Core simulation used in the thesis to analyze learning-like behavior
in EAP gel using a Pong-based task.
"""

import turtle
import math
import time
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----------------- GLOBAL SPEED CONTROL VARIABLE -----------------
SLOW_FACTOR = 1
# -----------------------------------------------------------------

# ----------------- NOISE CONTROL VARIABLE ------------------------
NOISE_FACTOR = 1.0
NOISE_BASELINE = 0.3
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
region_elapsed = {"A": 0.0, "B": 0.0, "C": 0.0}  # keeps cumulative on-time
# ------------------------------------------------

start_time = time.time()

score_display = turtle.Turtle(); score_display.hideturtle(); score_display.penup()
score_display.color("white"); score_display.goto(20, 350)

current_display = turtle.Turtle(); current_display.hideturtle(); current_display.penup()
current_display.color("black"); current_display.goto(350, 280)

# --- Data storage for graph ---
time_data, curr1_data, curr2_data, curr3_data = [], [], [], []

# -------------------- FUNCTIONS --------------------
BASELINE = 0.32
AMPLITUDE = 7.32
TAU = 123.47
MAX_SINE_AMPLITUDE = 0.35

# Each region has a different ON-STATE behavior
def f_A(t):
    return 2
    # return AMPLITUDE * math.exp(-t / TAU) + BASELINE  # exponential decay

def f_B(t):
    return 2
    # return AMPLITUDE * math.exp(-t / TAU) + BASELINE  # exponential decay


def f_C(t):
    return 2
    # return AMPLITUDE * math.exp(-t / TAU) + BASELINE  # exponential decay


region_funcs = {"A": f_A, "B": f_B, "C": f_C}

def sine_wave_noise(t, sensor_idx):
    amp_1 = 0.12 * math.sin(0.1 * t + sensor_idx * 1.5)
    amp_2 = 0.15 * math.sin(0.8 * t + sensor_idx * 0.5)
    amp_3 = 0.08 * math.sin(2.5 * t + sensor_idx * 0.9)
    sine_sum = amp_1 + amp_2 + amp_3
    normalized_sine = sine_sum / MAX_SINE_AMPLITUDE
    rand_comp = random.uniform(-0.05, 0.05) / MAX_SINE_AMPLITUDE
    return (normalized_sine + rand_comp) * NOISE_FACTOR

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

# ---------------- STEPWISE FUNCTION-BASED CURRENT MODEL ----------------
def compute_currents(y):
    """Each region follows its own defined function while ON.
       Memory is preserved between activations."""
    t_now = time.time() - start_time

    # Baseline noise for all sensors
    currents = np.array([
        sine_wave_noise(t_now, 0) + NOISE_BASELINE,
        sine_wave_noise(t_now, 1) + NOISE_BASELINE,
        sine_wave_noise(t_now, 2) + NOISE_BASELINE
    ])

    active = get_ball_region()
    idx_map = {"A": 0, "B": 1, "C": 2}

    for r in ["A", "B", "C"]:
        idx = idx_map[r]
        is_on = (r == active)
        f_region = region_funcs[r]

        if is_on and region_mem_value[r] is None:
            # First ever activation
            region_mem_value[r] = f_region(0.0)
            region_last_update[r] = t_now
            region_elapsed[r] = 0.0
            region_was_on[r] = True

        if is_on:
            # If region reactivated, reset timer difference but keep elapsed
            if not region_was_on[r]:
                region_last_update[r] = t_now
                region_was_on[r] = True

            # Time since last update
            dt = t_now - (region_last_update[r] or t_now)
            region_elapsed[r] += dt
            region_last_update[r] = t_now

            # Evaluate the region's unique function using cumulative ON time
            val = f_region(region_elapsed[r])
            region_mem_value[r] = val
            currents[idx] = val

        else:
            # Freeze memory when OFF
            region_was_on[r] = False

    return currents
# -----------------------------------------------------------------------

last_paddle_update = time.time()
paddle_update_dt = 0.1

def decide_paddle_y():
    y = ball.ycor()
    currents = compute_currents(y)
    t_now = time.time() - start_time
    time_data.append(t_now)
    curr1_data.append(currents[0]); curr2_data.append(currents[1]); curr3_data.append(currents[2])

    lowRangeC, upRangeC = -7.0, 7.6
    norm_currents = (currents - lowRangeC)/(upRangeC - lowRangeC)
    norm_currents = np.clip(norm_currents,0,1)

    x_positions = np.array([-1.0,0.0,1.0])
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

def update_current_display(currents, paddle_y):
    current_display.clear()

    elapsed = time.time() - start_time   # ⬅ NEW TIMER

    text = (
        f"Currents:\n"
        f"Top: {currents[0]:.3f} mA\n"
        f"Middle: {currents[1]:.3f} mA\n"
        f"Bottom: {currents[2]:.3f} mA\n"
        f"Paddle Y: {paddle_y:.1f}\n"
        f"Time: {elapsed:6.1f} s"   # ⬅ SHOW TIMER HERE
    )

    current_display.write(text, align="center", font=("Courier", 12, "bold"))

def plot_hit_rate():
    fig,axs=plt.subplots(3,1,figsize=(8,10))
    for i,r in enumerate(["A","B","C"]):
        hist,times=region_history[r],region_time[r]
        if hist: axs[i].plot(times,hist,marker="o",linewidth=2)
        axs[i].set_title(f"Region {r} Hit Rate vs Time")
        axs[i].set_xlabel("Time (s)"); axs[i].set_ylabel("Hit Rate")
        axs[i].set_ylim(0,1.05); axs[i].grid(True)
    plt.tight_layout(); plt.show()

def plot_currents_after_run():
    df=pd.DataFrame({"time":time_data,"current1":curr1_data,"current2":curr2_data,"current3":curr3_data})
    df.to_csv("currents_log.csv",index=False)
    print("✅ Current data saved to 'currents_log.csv'")
    plt.figure(figsize=(10,6))
    plt.plot(df["time"],df["current1"],label="Region A (Top, f_A=t)")
    plt.plot(df["time"],df["current2"],label="Region B (Middle, f_B=exp decay)")
    plt.plot(df["time"],df["current3"],label="Region C (Bottom, f_C=1/t)")
    plt.xlabel("Time (s)"); plt.ylabel("Current (mA)")
    plt.title("Independent Region Currents (Function-driven)")
    plt.legend(); plt.grid(True); plt.show()

def quit_game():
    win.bye(); plot_currents_after_run(); plot_hit_rate()

win.listen(); win.onkeypress(quit_game,"q")
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

        if ball.xcor()<-290:
            region_trials[region]+=1; trial_counter+=1
            current_t=time.time()-start_time
            for r in ["A","B","C"]:
                if region_trials[r]>0:
                    rate=region_hits[r]/region_trials[r]
                    region_history[r].append(rate)
                    region_time[r].append(current_t)
            current_score=0; ball.goto(0,0); initialize_ball_speed(); update_score()
        normalize_velocity()
