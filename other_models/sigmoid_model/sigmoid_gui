import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import time
import math
import threading
import tkinter as tk
from tkinter import ttk
import numpy as np


class SigmoidLearningSystem:
    def __init__(self):
        # 学習パラメータ
        self.stimulation_count = 0
        self.learning_rate = 0.15
        self.base_threshold = 50.0
        self.min_threshold = 8.0
        self.current_threshold = 50.0
        self.response_probability = 0.0

        # 忘却パラメータ
        self.decay_rate = 0.02
        self.last_input_time = time.time()
        self.forget_interval = 2.0

        # グラフ用データ（制限なしで全データを保存）
        self.all_time_data = []  # 全データを保存（制限なし）
        self.all_learning_data = []  # 全学習レベルを保存（制限なし）
        self.all_stimulation_data = []  # 全刺激回数を保存（制限なし）

        # リアルタイム表示用（制限あり）
        self.max_display_points = 200
        self.time_data = deque(maxlen=self.max_display_points)
        self.learning_level = deque(maxlen=self.max_display_points)
        self.input_events = []  # 入力イベントのタイミング

        # 統計
        self.total_inputs = 0
        self.recent_inputs = deque(maxlen=30)
        self.start_time = time.time()
        self.end_time = None

        # 制御フラグ
        self.running = True
        self.data_lock = threading.Lock()

        # GUI設定
        self.setup_gui()

        # 初期データポイント
        self.record_current_state()

        # 更新スレッド開始
        self.update_thread = threading.Thread(target=self.update_loop, daemon=True)
        self.update_thread.start()

    def setup_gui(self):
        """GUI設定"""
        self.root = tk.Tk()
        self.root.title("🧠 Sigmoid Learning System - Data Collection")
        self.root.geometry("500x350")
        self.root.configure(bg='#1a1a1a')

        # ウィンドウが閉じられた時の処理
        self.root.protocol("WM_DELETE_WINDOW", self.quit_application)

        # メインフレーム
        main_frame = tk.Frame(self.root, bg='#1a1a1a')
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)

        # タイトル
        title_label = tk.Label(main_frame, text="🧠 Sigmoid Learning System",
                               font=('Arial', 20, 'bold'), bg='#1a1a1a', fg='#00ff88')
        title_label.pack(pady=15)

        # 現在の学習レベル（大きく表示）
        self.level_frame = tk.Frame(main_frame, bg='#333333', relief='raised', bd=3)
        self.level_frame.pack(fill='x', pady=20)

        self.current_level_label = tk.Label(self.level_frame, text="Learning Level: 0.0%",
                                            font=('Arial', 24, 'bold'), bg='#333333', fg='#00ff88')
        self.current_level_label.pack(pady=15)

        # 統計フレーム
        stats_frame = tk.LabelFrame(main_frame, text="📊 Real-time Statistics",
                                    font=('Arial', 12, 'bold'), bg='#1a1a1a', fg='white')
        stats_frame.pack(fill='x', pady=15)

        # 統計ラベル
        self.stats_labels = {}
        stats_info = [
            ("Total Stimulations:", "total_inputs", "#00aaff"),
            ("Response Probability:", "response_prob", "#ff6600"),
            ("Current Threshold:", "threshold", "#aa00ff"),
            ("Input Rate (per min):", "input_rate", "#ffaa00"),
        ]

        for i, (label_text, key, color) in enumerate(stats_info):
            row_frame = tk.Frame(stats_frame, bg='#1a1a1a')
            row_frame.pack(fill='x', pady=3, padx=10)

            label = tk.Label(row_frame, text=label_text, font=('Arial', 11),
                             bg='#1a1a1a', fg='#cccccc', anchor='w')
            label.pack(side='left')

            value_label = tk.Label(row_frame, text="0", font=('Arial', 11, 'bold'),
                                   bg='#1a1a1a', fg=color, anchor='e')
            value_label.pack(side='right')
            self.stats_labels[key] = value_label

        # 操作説明
        instruction_frame = tk.Frame(main_frame, bg='#2a2a2a', relief='sunken', bd=2)
        instruction_frame.pack(fill='x', pady=15)

        instruction_label = tk.Label(instruction_frame,
                                     text="📌 Press SPACE to stimulate learning\n📈 Close window to see full session graph!",
                                     font=('Arial', 12), bg='#2a2a2a', fg='#cccccc', justify='center')
        instruction_label.pack(pady=10)

        # ボタン
        button_frame = tk.Frame(main_frame, bg='#1a1a1a')
        button_frame.pack(fill='x', pady=10)

        reset_button = tk.Button(button_frame, text="🔄 Reset", command=self.reset_system,
                                 font=('Arial', 12), bg='#ff6600', fg='white', width=12)
        reset_button.pack(side='left', padx=5)

        quit_button = tk.Button(button_frame, text="❌ Quit & Show Full Graph", command=self.quit_application,
                                font=('Arial', 12), bg='#ff4444', fg='white', width=18)
        quit_button.pack(side='right', padx=5)

        # キーバインド
        self.root.bind('<KeyPress-space>', self.on_space_press)
        self.root.bind('<KeyPress-r>', lambda e: self.reset_system())
        self.root.focus_set()

    def calculate_sigmoid_learning(self, stimulation_count):
        """シグモイド学習関数"""
        learning_progress = stimulation_count * self.learning_rate
        sigmoid_factor = 1 / (1 + math.exp(-learning_progress + 5))

        # 閾値の計算
        threshold_reduction = (self.base_threshold - self.min_threshold) * sigmoid_factor
        new_threshold = self.base_threshold - threshold_reduction

        # 応答確率の計算
        response_probability = sigmoid_factor * 0.95

        # 学習レベル（0-100%）
        learning_level = sigmoid_factor * 100

        return new_threshold, response_probability, learning_level

    def apply_forgetting(self):
        """忘却の適用"""
        current_time = time.time()
        time_since_input = current_time - self.last_input_time

        if time_since_input > self.forget_interval:
            forget_amount = self.decay_rate * (time_since_input - self.forget_interval)
            self.stimulation_count = max(0, self.stimulation_count - forget_amount)
            self.current_threshold, self.response_probability, _ = self.calculate_sigmoid_learning(
                self.stimulation_count)

    def record_current_state(self):
        """現在の状態をデータに記録"""
        current_time = time.time()
        elapsed_time = current_time - self.start_time

        _, _, learning_level = self.calculate_sigmoid_learning(self.stimulation_count)

        with self.data_lock:
            # リアルタイム表示用（制限あり）
            self.time_data.append(elapsed_time)
            self.learning_level.append(learning_level)

            # 全データ保存用（制限なし）- これが最終グラフで使用される
            self.all_time_data.append(elapsed_time)
            self.all_learning_data.append(learning_level)
            self.all_stimulation_data.append(self.stimulation_count)

    def on_space_press(self, event):
        """スペースキー押下時の処理"""
        current_time = time.time()

        # 刺激回数増加
        self.stimulation_count += 1
        self.total_inputs += 1
        self.last_input_time = current_time

        # イベント記録
        elapsed_time = current_time - self.start_time
        self.input_events.append(elapsed_time)
        self.recent_inputs.append(current_time)

        # 学習パラメータの更新
        self.current_threshold, self.response_probability, current_learning = self.calculate_sigmoid_learning(
            self.stimulation_count)

        # データ記録
        self.record_current_state()

        print(f"🔥 Stimulation #{self.total_inputs}! Learning: {current_learning:.1f}%")

        # GUI即時更新をメインスレッドで実行
        self.root.after(0, self.update_gui_immediate, current_learning)

    def update_gui_immediate(self, current_learning):
        """GUI即時更新（メインスレッドで実行）"""
        try:
            self.current_level_label.config(text=f"Learning Level: {current_learning:.1f}%")
        except Exception as e:
            print(f"GUI update error: {e}")

    def update_gui_stats(self):
        """GUI統計の更新"""
        if not self.running:
            return

        try:
            current_time = time.time()
            elapsed_time = current_time - self.start_time

            # 入力レート計算
            if elapsed_time > 0:
                inputs_per_minute = (self.total_inputs / elapsed_time) * 60
            else:
                inputs_per_minute = 0

            # 現在の学習レベル
            _, _, current_learning = self.calculate_sigmoid_learning(self.stimulation_count)

            # GUI更新をメインスレッドで実行
            self.root.after(0, self._update_gui_labels, current_learning, inputs_per_minute)

        except Exception as e:
            print(f"Stats update error: {e}")

    def _update_gui_labels(self, current_learning, inputs_per_minute):
        """GUI ラベルの更新（メインスレッドで実行）"""
        try:
            self.stats_labels["total_inputs"].config(text=f"{self.total_inputs}")
            self.stats_labels["response_prob"].config(text=f"{self.response_probability:.3f}")
            self.stats_labels["threshold"].config(text=f"{self.current_threshold:.1f}")
            self.stats_labels["input_rate"].config(text=f"{inputs_per_minute:.1f}")

            # メインレベル表示
            self.current_level_label.config(text=f"Learning Level: {current_learning:.1f}%")
        except Exception as e:
            print(f"Label update error: {e}")

    def update_loop(self):
        """メインアップデートループ"""
        while self.running:
            try:
                # 忘却の適用
                self.apply_forgetting()

                # 定期的なデータ記録（忘却の様子も記録）
                self.record_current_state()

                # GUI統計の更新
                self.update_gui_stats()

                time.sleep(0.2)  # 200ms間隔で更新（負荷軽減）

            except Exception as e:
                print(f"Update loop error: {e}")
                time.sleep(0.5)

    def reset_system(self):
        """システムリセット"""
        with self.data_lock:
            self.stimulation_count = 0
            self.total_inputs = 0
            self.current_threshold = self.base_threshold
            self.response_probability = 0.0
            self.last_input_time = time.time()
            self.start_time = time.time()
            self.end_time = None

            # 全データクリア
            self.all_time_data.clear()
            self.all_learning_data.clear()
            self.all_stimulation_data.clear()

            # リアルタイムデータクリア
            self.time_data.clear()
            self.learning_level.clear()
            self.input_events.clear()
            self.recent_inputs.clear()

        # 初期データポイントを再記録
        self.record_current_state()
        print("🔄 System Reset! Full session data cleared.")

    def show_final_graph(self):
        """GUI終了後の最終グラフ表示（セッション全体のデータを使用）"""
        print("📈 Generating complete session analysis graph...")

        # 全データを使用（制限なし）
        with self.data_lock:
            time_list = list(self.all_time_data)
            learning_list = list(self.all_learning_data)
            stimulation_list = list(self.all_stimulation_data)
            events_list = list(self.input_events)

        # セッション全体の時間を計算
        total_session_time = self.end_time - self.start_time if self.end_time else 0

        # データが少ない場合の処理
        if len(time_list) < 2:
            print("⚠️ Not enough data collected. Please run the system longer next time.")
            return

        print(f"📊 Displaying full session data: {len(time_list)} data points from t=0 to t={total_session_time:.1f}s")

        # グラフ作成
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

        # メインタイトル
        fig.suptitle(f'🧠 Complete Sigmoid Learning Session Analysis\n'
                     f'Total Duration: {total_session_time:.1f}s | Total Inputs: {self.total_inputs} | '
                     f'Data Points: {len(time_list)} | Final Learning: {learning_list[-1]:.1f}%',
                     fontsize=16, color='white', y=0.95)

        # 上部グラフ: 学習レベルの推移
        ax1.plot(time_list, learning_list, '-', color='#00ff88', linewidth=2,
                 label='Learning Level (%)', alpha=0.9)

        # 入力イベントのマーカー
        for event_time in events_list:
            if event_time <= max(time_list):  # 範囲内のイベントのみ
                # イベント時の学習レベルを補間で求める
                event_learning = np.interp(event_time, time_list, learning_list)
                ax1.plot(event_time, event_learning, 'o', color='red', markersize=6, alpha=0.7)

        # 最初と最後のポイントを強調
        if time_list and learning_list:
            ax1.plot(time_list[0], learning_list[0], 'o', color='blue',
                     markersize=10, label=f'Start ({learning_list[0]:.1f}%)',
                     markeredgecolor='white', markeredgewidth=2)
            ax1.plot(time_list[-1], learning_list[-1], 'o', color='red',
                     markersize=12, label=f'End ({learning_list[-1]:.1f}%)',
                     markeredgecolor='white', markeredgewidth=2)

        # 学習フェーズの領域分け
        ax1.axhspan(0, 25, alpha=0.1, color='blue', label='Initial (0-25%)')
        ax1.axhspan(25, 75, alpha=0.1, color='yellow', label='Growth (25-75%)')
        ax1.axhspan(75, 100, alpha=0.1, color='green', label='Maturity (75-100%)')

        ax1.set_xlim(0, max(total_session_time, max(time_list) if time_list else 1))
        ax1.set_ylim(-5, 105)
        ax1.set_ylabel('Learning Level (%)', color='white', fontsize=12)
        ax1.set_title('Learning Progress Over Complete Session', color='white', fontsize=14)
        ax1.grid(True, alpha=0.3, color='gray')
        ax1.legend(loc='center right', fontsize=10)

        # 下部グラフ: 刺激回数の推移
        ax2.plot(time_list, stimulation_list, '-', color='#ffaa00', linewidth=2,
                 label='Stimulation Count', alpha=0.9)

        # 入力イベントの垂直線
        for event_time in events_list:
            if event_time <= max(time_list):
                ax2.axvline(x=event_time, color='red', alpha=0.3, linewidth=1)

        ax2.set_xlim(0, max(total_session_time, max(time_list) if time_list else 1))
        ax2.set_ylim(0, max(stimulation_list) * 1.1 if stimulation_list else 1)
        ax2.set_xlabel('Time from Start (seconds)', color='white', fontsize=12)
        ax2.set_ylabel('Stimulation Count', color='white', fontsize=12)
        ax2.set_title('Stimulation History', color='white', fontsize=14)
        ax2.grid(True, alpha=0.3, color='gray')
        ax2.legend(loc='upper left', fontsize=10)

        # 詳細統計情報
        if time_list and learning_list:
            stats_text = (
                f'Complete Session Analysis:\n'
                f'• Total Duration: {total_session_time:.1f} seconds\n'
                f'• Data Points Collected: {len(time_list)}\n'
                f'• Total Stimulations: {self.total_inputs}\n'
                f'• Average Input Rate: {(self.total_inputs / total_session_time) * 60:.1f}/min\n'
                f'• Initial Learning: {learning_list[0]:.1f}%\n'
                f'• Final Learning: {learning_list[-1]:.1f}%\n'
                f'• Net Learning Gain: +{learning_list[-1] - learning_list[0]:.1f}%\n'
                f'• Peak Learning: {max(learning_list):.1f}%\n'
                f'• Learning Efficiency: {(learning_list[-1] / max(1, self.total_inputs)) * 100:.1f}%/input'
            )

            ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
                     fontsize=10, verticalalignment='top', horizontalalignment='left',
                     bbox=dict(boxstyle='round', facecolor='black', alpha=0.8, edgecolor='white'),
                     color='white')

        # レイアウト調整
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)

        print("📊 Complete session graph generated! This shows ALL data from start to finish.")
        print(f"   - Session ran for {total_session_time:.1f} seconds")
        print(f"   - Captured {len(time_list)} data points")
        print(f"   - Shows learning progression from {learning_list[0]:.1f}% to {learning_list[-1]:.1f}%")
        plt.show()

    def quit_application(self):
        """アプリケーション終了"""
        print("🛑 Shutting down GUI and preparing complete session analysis...")

        # 終了時間を記録
        self.end_time = time.time()
        self.running = False

        # 最終データポイントを確実に記録
        self.record_current_state()

        # GUIを閉じる
        try:
            if hasattr(self, 'root'):
                self.root.quit()
                self.root.destroy()
        except:
            pass

        # 少し待ってから最終グラフを表示
        time.sleep(0.5)
        self.show_final_graph()

    def run(self):
        """アプリケーション実行"""
        try:
            print("🚀 Sigmoid Learning System - Complete Session Data Collection!")
            print("📝 Instructions:")
            print("   - Press SPACE in the GUI window to stimulate learning")
            print("   - Watch the real-time statistics in the GUI")
            print("   - Press 'Quit & Show Full Graph' to see COMPLETE session analysis")
            print("   - Press R to reset system")
            print("\n🎯 Try different input patterns:")
            print("   - Rapid inputs for fast learning")
            print("   - Pauses to see forgetting in action")
            print("   - Mixed patterns for complex learning curves")
            print("\n📊 The final graph will show ALL data from t=0 to session end!")

            # GUIメインループ
            self.root.mainloop()

        except KeyboardInterrupt:
            self.quit_application()
        except Exception as e:
            print(f"Application error: {e}")
            self.quit_application()


if __name__ == "__main__":
    print("🧠 Starting Sigmoid Learning System - Complete Session Analysis...")
    try:
        app = SigmoidLearningSystem()
        app.run()
    except Exception as e:
        print(f"Failed to start application: {e}")
        print("Please ensure you have matplotlib, numpy, and tkinter installed:")
        print("pip install matplotlib numpy")
