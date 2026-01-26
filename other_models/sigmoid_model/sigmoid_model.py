import turtle
import math
import time
import random

# Set up the window
win = turtle.Screen()
win.title("Brain Gel Learning Simulation - Realistic Learning Curve")
win.bgcolor("#87CEFA")
win.setup(width=600, height=900)
win.tracer(0)

# Define block size
block_width = 300
block_height = 300

# Paddle
paddle = turtle.Turtle()
paddle.speed(0)
paddle.shape("square")
paddle.color("white")
paddle.shapesize(stretch_wid=(block_height / 20), stretch_len=1)
paddle.penup()
paddle.goto(-290, 0)

# Ball
ball = turtle.Turtle()
ball.speed(0)
ball.shape("square")
ball.color("white")
ball.penup()
ball.goto(0, 0)
ball.dx = 3.0
ball.dy = 3.0

# Drawer for highlighting
highlight = turtle.Turtle()
highlight.speed(0)
highlight.penup()
highlight.hideturtle()
highlight.pensize(0)

# Grid lines drawer
grid = turtle.Turtle()
grid.speed(0)
grid.penup()
grid.hideturtle()
grid.pensize(3)

# Score
score = 0
misses = 0
score_display = turtle.Turtle()
score_display.speed(0)
score_display.color("white")
score_display.penup()
score_display.hideturtle()
score_display.goto(10, 320)

# Learning parameters - バランス調整済み
gel_regions = {
    "A": {
        "current": 0.0,
        "threshold": 60.0,  # 調整済み初期閾値
        "base_threshold": 60.0,
        "min_threshold": 12.0,  # 調整済み最小閾値
        "stimulation_count": 0,
        "successful_responses": 0,
        "learning_rate": 0.08,  # 調整済み学習率
        "current_decay": 0.95,  # 調整済み減衰
        "last_stimulus_time": 0.0,
        "response_probability": 0.0,
        "is_responding": False,
        "response_duration": 0.0,
        "max_response_duration": 1.0,
        "last_count_time": None,
        "refractory_period": 0.0,
        "max_refractory": 1.0  # 短縮した不応期
    },
    "B": {
        "current": 0.0,
        "threshold": 60.0,
        "base_threshold": 60.0,
        "min_threshold": 12.0,
        "stimulation_count": 0,
        "successful_responses": 0,
        "learning_rate": 0.08,
        "current_decay": 0.95,
        "last_stimulus_time": 0.0,
        "response_probability": 0.0,
        "is_responding": False,
        "response_duration": 0.0,
        "max_response_duration": 1.0,
        "last_count_time": None,
        "refractory_period": 0.0,
        "max_refractory": 1.0
    },
    "C": {
        "current": 0.0,
        "threshold": 60.0,
        "base_threshold": 60.0,
        "min_threshold": 12.0,
        "stimulation_count": 0,
        "successful_responses": 0,
        "learning_rate": 0.08,
        "current_decay": 0.95,
        "last_stimulus_time": 0.0,
        "response_probability": 0.0,
        "is_responding": False,
        "response_duration": 0.0,
        "max_response_duration": 1.0,
        "last_count_time": None,
        "refractory_period": 0.0,
        "max_refractory": 1.0
    }
}

# パドル制御パラメータ
paddle_is_active = False
active_region = None
paddle_target_y = 0.0
paddle_speed = 4.0
stimulus_current = 6.0  # バランス調整済み刺激
dt = 0.1

# 領域の境界定義（Y座標）
REGION_BOUNDARIES = {
    "A": {"center": 300, "min": 150, "max": 450},
    "B": {"center": 0, "min": -150, "max": 150},
    "C": {"center": -300, "min": -450, "max": -150}
}

# 統計追跡
total_stimulations = 0
total_hits = 0
hit_rate = 0.0
start_time = time.time()


def draw_regions():
    grid.clear()
    grid.color("black")
    for x in [-300, 0, 300]:
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
    col = 0 if ball.xcor() < 0 else 1
    row = 0 if ball.ycor() > 150 else 1 if ball.ycor() > -150 else 2
    x = -300 + col * 300
    y = 450 - row * 300
    highlight.clear()
    highlight.goto(x, y)
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


def get_ball_region():
    if ball.ycor() > 150:
        return "A"
    elif ball.ycor() > -150:
        return "B"
    else:
        return "C"


def calculate_learning_curve(stimulation_count, learning_rate, base_threshold, min_threshold):
    """より現実的なS字カーブ学習関数"""
    # 学習進行度の計算
    learning_progress = stimulation_count * learning_rate

    # シグモイド関数（バランス調整済み）
    sigmoid_factor = 1 / (1 + math.exp(-learning_progress + 6))  # 中央値を6に調整

    # 閾値の減少
    threshold_reduction = (base_threshold - min_threshold) * sigmoid_factor
    new_threshold = base_threshold - threshold_reduction

    # 応答確率の計算（段階的な向上）
    response_probability = sigmoid_factor * 0.85  # 最大85%の応答確率

    return new_threshold, response_probability


def update_gel_system():
    global paddle_is_active, active_region, total_stimulations

    current_time = time.time()
    ball_region = get_ball_region()

    # 全領域の状態更新
    for region_name, gel in gel_regions.items():
        # 不応期の処理
        if gel["refractory_period"] > 0:
            gel["refractory_period"] -= dt
            if gel["refractory_period"] <= 0:
                gel["refractory_period"] = 0

        # 電流の自然減衰（不応期中でも減衰）
        gel["current"] *= gel["current_decay"]

        # 応答持続時間の管理
        if gel["is_responding"]:
            gel["response_duration"] += dt
            if gel["response_duration"] >= gel["max_response_duration"]:
                gel["is_responding"] = False
                gel["response_duration"] = 0.0

    # 現在の領域への刺激
    current_gel = gel_regions[ball_region]

    # 不応期中でなければ刺激を蓄積
    if current_gel["refractory_period"] <= 0:
        current_gel["current"] += stimulus_current

    current_gel["last_stimulus_time"] = current_time

    # 刺激回数の更新（より頻繁な更新）
    if current_gel["last_count_time"] is None:
        current_gel["last_count_time"] = current_time
    elif current_time - current_gel["last_count_time"] > 0.8:  # 0.8秒間隔に調整
        current_gel["stimulation_count"] += 1
        current_gel["last_count_time"] = current_time
        total_stimulations += 1

        # 学習曲線の計算
        new_threshold, response_prob = calculate_learning_curve(
            current_gel["stimulation_count"],
            current_gel["learning_rate"],
            current_gel["base_threshold"],
            current_gel["min_threshold"]
        )

        current_gel["threshold"] = new_threshold
        current_gel["response_probability"] = response_prob

    # 応答の判定（不応期中は応答しない）
    if (current_gel["refractory_period"] <= 0 and
            current_gel["current"] >= current_gel["threshold"]):

        # 確率的応答判定
        if random.random() < current_gel["response_probability"]:
            if not current_gel["is_responding"]:
                # 応答開始
                current_gel["is_responding"] = True
                current_gel["response_duration"] = 0.0
                current_gel["successful_responses"] += 1
                current_gel["current"] = 0.0  # 応答後リセット
                current_gel["refractory_period"] = current_gel["max_refractory"]  # 不応期開始

                # パドル制御の活性化
                paddle_is_active = True
                active_region = ball_region


def move_paddle_intelligently():
    global paddle_target_y

    if paddle_is_active and active_region:
        region_bounds = REGION_BOUNDARIES[active_region]
        region_center = region_bounds["center"]
        region_min = region_bounds["min"]
        region_max = region_bounds["max"]

        ball_y = ball.ycor()
        paddle_target_y = ball_y

        paddle_half_height = 150
        effective_min = region_min + paddle_half_height
        effective_max = region_max - paddle_half_height

        paddle_target_y = max(effective_min, min(effective_max, paddle_target_y))
    else:
        paddle_target_y = 0

    # 滑らかな移動
    current_y = paddle.ycor()
    y_diff = paddle_target_y - current_y
    move_speed = min(paddle_speed, abs(y_diff) * 0.1 + 1.0)

    if abs(y_diff) > 0.5:
        direction = 1 if y_diff > 0 else -1
        new_y = current_y + direction * move_speed
        new_y = max(-300, min(300, new_y))
        paddle.sety(new_y)


def normalize_velocity():
    """ボールの速度を一定に保つ"""
    current_speed = math.sqrt(ball.dx ** 2 + ball.dy ** 2)
    target_speed = 3.0

    if current_speed > 0:
        scale = target_speed / current_speed
        ball.dx *= scale
        ball.dy *= scale


def update_score():
    global hit_rate
    if total_stimulations > 0:
        hit_rate = (total_hits / total_stimulations) * 100

    score_display.clear()
    score_display.goto(15, 370)
    score_display.write(f"Score: {score}", align="left", font=("Courier", 16, "normal"))
    score_display.goto(15, 350)
    score_display.write(f"Misses: {misses}", align="left", font=("Courier", 16, "normal"))
    score_display.goto(15, 330)
    score_display.write(f"Total Stimulations: {total_stimulations}", align="left", font=("Courier", 16, "normal"))

    # 各領域の詳細な学習状態表示
    y_pos = 280
    for region_name, gel in gel_regions.items():
        score_display.goto(15, y_pos)
        score_display.write(f"Region {region_name}:", align="left", font=("Courier", 12, "bold"))
        y_pos -= 15
        score_display.goto(25, y_pos)
        score_display.write(f"Current: {gel['current']:.1f}, Threshold: {gel['threshold']:.1f}",
                            align="left", font=("Courier", 10, "normal"))
        y_pos -= 12
        score_display.goto(25, y_pos)
        score_display.write(f"Stimulations: {gel['stimulation_count']}, Prob: {gel['response_probability']:.2f}",
                            align="left", font=("Courier", 10, "normal"))
        y_pos -= 12
        score_display.goto(25, y_pos)
        active_status = "ACTIVE" if (active_region == region_name and paddle_is_active) else "inactive"
        refractory_status = f"Refractory: {gel['refractory_period']:.1f}s" if gel['refractory_period'] > 0 else "Ready"
        score_display.write(f"Status: {active_status}, {refractory_status}",
                            align="left", font=("Courier", 10, "normal"))
        y_pos -= 20


def quit_game():
    win.bye()


win.listen()
win.onkeypress(quit_game, "q")

# 初期化
draw_regions()
update_score()

# メインゲームループ
while True:
    win.update()

    ball.setx(ball.xcor() + ball.dx)
    ball.sety(ball.ycor() + ball.dy)

    highlight_region()
    update_gel_system()
    move_paddle_intelligently()

    # ボールの衝突判定
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

    # パドルとの衝突
    if -290 < ball.xcor() < -260:
        if paddle.ycor() - 150 < ball.ycor() < paddle.ycor() + 150:
            ball.setx(-260)
            ball.dx *= -1
            normalize_velocity()
            score += 1
            total_hits += 1
            update_score()

    # ミス
    if ball.xcor() < -290:
        ball.setx(-290)
        ball.dx *= -1
        normalize_velocity()
        misses += 1
        paddle_is_active = False
        active_region = None
        update_score()

    normalize_velocity()
