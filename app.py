import os
import time
import requests
import json
import random
import socket
from queue import Queue, Empty
from flask import Flask, jsonify, render_template
from threading import Thread, Event, Lock
import google.generativeai as genai
import keras
import numpy as np
from datetime import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import cv2

# Keras 모델 로드
MODEL_PATH = r"C:\Users\HyejinPark\Desktop\trained10.keras"
try:
    LOADED_MODEL = keras.models.load_model(MODEL_PATH)
    # 모델의 입력 형태를 확인합니다. 예: (None, 224, 224, 3)
    # if LOADED_MODEL:
    #     print(f"모델 입력 형태: {LOADED_MODEL.input_shape}")
except Exception as e:
    print(f"모델 로드 중 오류 발생: {e}")
    print("임시로 무작위 상태를 반환합니다. 모델 경로를 확인하세요.")
    LOADED_MODEL = None

STATES = ["앉기", "서기", "걷기", "없기"]

# Gemini API 키 설정 (환경 변수 또는 직접 설정)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("API 키가 설정되지 않았습니다. 환경 변수 'GEMINI_API_KEY'를 설정해주세요.")
    # exit() # 개발 편의를 위해 임시로 주석 처리

genai.configure(api_key=GEMINI_API_KEY)

# Gemini 모델 설정
model = genai.GenerativeModel('gemini-1.5-flash-latest')

app = Flask(__name__)

# 전역 변수로 사용자 상태와 시간 추적
user_state = "없기"  # 초기 상태
state_start_time = time.time()
message_sent = False  # 메시지 전송 여부

# CSI 데이터를 저장할 큐
csi_data_queue = Queue()

# 활동 시간 누적 및 스레드 동기화를 위한 변수
user_state_durations = {state: 0 for state in STATES}
durations_lock = Lock()

# 상태 추적 스레드 제어
stop_event = Event()

# 전처리용 히트맵 데이터 버퍼
realtime_heatmap_buffer = []

def csi_receiver():
    """
    UDP 소켓을 열고 실시간 CSI 데이터를 수신하여 큐에 저장합니다.
    """
    HOST = '0.0.0.0'  # 모든 IP 주소에서 수신
    PORT = 5500
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_address = (HOST, PORT)
    sock.bind(server_address)
    
    print(f"UDP 소켓이 {HOST}:{PORT}에서 수신 대기 중...")
    
    while not stop_event.is_set():
        try:
            data, address = sock.recvfrom(1024 * 64) # 64KB 버퍼
            
            # 받은 pcap 파일의 바이트 데이터를 NumPy 배열로 변환
            # (20, 10)의 형태를 가진 CSI 데이터로 가정
            received_data = np.frombuffer(data, dtype=np.float32).reshape(20, 10)
            
            # 큐에 데이터 저장 (가장 최신 데이터만 유지)
            if not csi_data_queue.empty():
                try:
                    csi_data_queue.get_nowait()
                except Empty:
                    pass
            csi_data_queue.put(received_data)
        except Exception as e:
            print(f"데이터 수신 중 오류 발생: {e}")
            break
    sock.close()

def get_real_time_csi_data():
    """
    큐에서 가장 최신 CSI 데이터를 가져옵니다.
    """
    if not csi_data_queue.empty():
        return csi_data_queue.get_nowait()
    return None

def preprocess_csi(raw_csi_data):
    """
    실시간 CSI 데이터를 히트맵 이미지 데이터로 변환합니다.
    """
    global realtime_heatmap_buffer

    if raw_csi_data is None:
        return None
        
    # 1. CSI 진폭을 dB 값으로 변환
    amplitudes = np.abs(raw_csi_data)
    db_values = 20 * np.log10(np.where(amplitudes > 0, amplitudes, np.nan))
    db_values = np.where(db_values > 70, 55.94, db_values)
    
    # 2. 데이터 버퍼에 추가
    realtime_heatmap_buffer.append(db_values)

    # 3. 버퍼가 20개 쌓이면 히트맵 생성
    if len(realtime_heatmap_buffer) < 20:
        return None
    
    # 4. 히트맵 생성 (인-메모리)
    data = pd.DataFrame(np.array(realtime_heatmap_buffer))

    plt.figure(figsize=(2.24, 2.24), dpi=100)
    sns.heatmap(data, cmap='coolwarm', cbar=False, xticklabels=False, yticklabels=False)
    plt.axis('off')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    buf.seek(0)
    
    # 5. 이미지를 NumPy 배열로 변환
    img_data = plt.imread(buf, format='png')
    
    # 6. CNN 모델 입력 형태에 맞게 최종 전처리
    if img_data.shape[2] == 4:
        img_data = img_data[:, :, :3]
    
    # 모델에 맞는 크기 (1, 224, 224, 3)로 차원 확장
    processed_image = np.expand_dims(img_data, axis=0)

    # 7. 버퍼 초기화
    realtime_heatmap_buffer = []

    return processed_image

def classify_user_state():
    if LOADED_MODEL is None:
        return random.choice(STATES)
    
    try:
        csi_data = get_real_time_csi_data()
        if csi_data is None:
            return "데이터 없음"
        
        processed_data = preprocess_csi(csi_data)
        
        if processed_data is None:
            return "데이터 없음"
            
        prediction = LOADED_MODEL.predict(processed_data)
        
        return STATES[np.argmax(prediction)]
    except Exception as e:
        print(f"모델 예측 중 오류 발생: {e}")
        return "데이터 없음"

def get_gemini_guidance(user_state, user_state_duration_minutes):
    prompt = ""
    if user_state == "앉기":
        prompt = f"사용자는 현재 {user_state_duration_minutes:.0f}분 동안 앉아있었습니다. 앉아있는 시간을 줄이기 위한 친절한 건강 가이드와 간단한 스트레칭을 50자 내외로 제공해줘."
    elif user_state == "서기":
        prompt = f"사용자는 현재 {user_state_duration_minutes:.0f}분 동안 서 있었습니다. 다리 건강을 위한 친절한 가이드와 앉아서 할 수 있는 스트레칭을 50자 내외로 제공해줘."
    elif user_state == "걷기":
        prompt = f"사용자는 현재 {user_state_duration_minutes:.0f}분 동안 걷고 있었습니다. 걷기 운동의 장점과 올바른 자세에 대한 조언을 50자 내외로 제공해줘."
    elif user_state == "없기":
        return "현재 방에 사람이 없습니다. 돌아오면 건강 가이드를 다시 시작합니다."
    elif user_state == "데이터 없음":
        return "데이터 수신에 문제가 발생했습니다. Raspberry Pi의 상태를 확인해 주세요."

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Gemini API 호출 중 오류 발생: {e}")
        return "건강 가이드를 생성할 수 없습니다. 잠시 후 다시 시도해 주세요."

def check_user_state_and_send_message():
    global user_state, state_start_time, message_sent
    while not stop_event.is_set():
        new_state = classify_user_state()
        
        if new_state != user_state:
            duration_to_add = (time.time() - state_start_time)
            with durations_lock:
                if user_state in user_state_durations:
                    user_state_durations[user_state] += duration_to_add
            
            user_state = new_state
            state_start_time = time.time()
            message_sent = False
            print(f"상태 변경: {user_state}")
        
        current_duration_seconds = time.time() - state_start_time
        if current_duration_seconds >= 30 and not message_sent:
            duration_minutes = current_duration_seconds / 60
            guidance = get_gemini_guidance(user_state, duration_minutes)
            print(f"사용자에게 메시지 전송: {guidance}")
            message_sent = True
            
        time.sleep(1)

@app.route('/daily_summary')
def get_daily_summary():
    with durations_lock:
        sitting_time_minutes = int(user_state_durations.get("앉기", 0) / 60)
        standing_time_minutes = int(user_state_durations.get("서기", 0) / 60)
        walking_time_minutes = int(user_state_durations.get("걷기", 0) / 60)
        
        current_state_duration = int((time.time() - state_start_time) / 60)
        if user_state == "앉기":
            sitting_time_minutes += current_state_duration
        elif user_state == "서기":
            standing_time_minutes += current_state_duration
        elif user_state == "걷기":
            walking_time_minutes += current_state_duration
            
    total_time_minutes = sitting_time_minutes + standing_time_minutes + walking_time_minutes
    active_time_minutes = standing_time_minutes + walking_time_minutes
    
    daily_goal_minutes = 90
    goal_percentage = int((active_time_minutes / daily_goal_minutes) * 100)

    return jsonify({
        "total_time": f"{total_time_minutes}분",
        "sitting_time": f"{sitting_time_minutes}분",
        "standing_time": f"{standing_time_minutes}분",
        "walking_time": f"{walking_time_minutes}분",
        "goal_percentage": goal_percentage
    })

@app.route('/activity_history')
def get_activity_history():
    history = []
    days = ["일", "월", "화", "수", "목", "금", "토"]
    for i in range(7):
        day = days[i]
        walking = random.randint(30, 90)
        standing = random.randint(120, 300)
        sitting = random.randint(15, 60)
        history.append({
            "day": day,
            "walking_minutes": walking,
            "standing_minutes": standing,
            "sitting_minutes": sitting
        })
    return jsonify(history)

@app.route('/hourly_activity_summary')
def get_hourly_activity_summary():
    hourly_summary = []
    for hour in range(24):
        walking = random.randint(0, 10)
        standing = random.randint(0, 20)
        sitting = 60 - walking - standing if (walking + standing) <= 60 else 0
        hourly_summary.append({
            "hour": f"{hour}시",
            "walking_minutes": walking,
            "standing_minutes": standing,
            "sitting_minutes": sitting
        })
    return jsonify(hourly_summary)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/status')
def get_status():
    duration = time.time() - state_start_time
    guidance = ""
    if message_sent:
        duration_minutes = duration / 60
        guidance = get_gemini_guidance(user_state, duration_minutes)
    return jsonify({
        "status": user_state,
        "duration": f"{int(duration / 60)}분 {int(duration % 60)}초",
        "message_sent": message_sent,
        "guidance": guidance
    })

if __name__ == '__main__':
    receiver_thread = Thread(target=csi_receiver)
    receiver_thread.daemon = True
    receiver_thread.start()
    
    state_thread = Thread(target=check_user_state_and_send_message)
    state_thread.daemon = True
    state_thread.start()
    
    app.run(debug=True, use_reloader=False)
