# -*- coding: utf-8 -*-
import subprocess
import time
import socket
import os

# NOTE: 이 코드는 라즈베리파이에서 CSI 데이터를 pcap 파일로 캡처하여 PC로 전송하는 예시입니다.

# 데이터 전송 간격 (초 단위)
# 캡처 시간과 동일하게 설정하세요.
CAPTURE_INTERVAL_SECONDS = 1

# 캡처할 패킷 수
PACKET_COUNT = 20

# PC의 IP 주소와 포트
HOST = '192.168.137.1'  # PC의 IP 주소로 변경하세요 (예: '192.168.0.5')
PORT = 5500

# UDP 소켓 생성
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
server_address = (HOST, PORT)

print("CSI 패킷 캡처 및 전송을 시작합니다. PC의 {}:{}로 전송됩니다.".format(HOST, PORT))

try:
    while True:
        # pcap 파일 이름 설정 (타임스탬프 활용)
        pcap_filename = f"capture_{int(time.time())}.pcap"
        
        # tcpdump 명령어를 사용하여 CSI 패킷 캡처
        # -i: 인터페이스 (wlan0)
        # -vv: 상세 정보 출력
        # -w: 파일로 저장
        # -c: 캡처할 패킷 수
        # udp port 5500: 특정 포트로 들어오는 UDP 패킷만 캡처
        command = f"sudo tcpdump -i wlan0 udp port 5500 -vv -w {pcap_filename} -c {PACKET_COUNT}"
        print(f"명령어 실행: {command}")
        
        # subprocess를 사용하여 명령어 실행
        subprocess.run(command, shell=True, check=True)
        
        # 캡처된 pcap 파일을 읽어서 바이트 형태로 전송
        try:
            with open(pcap_filename, 'rb') as f:
                pcap_data = f.read()
                sock.sendto(pcap_data, server_address)
            
            print(f"파일 '{pcap_filename}'을 PC로 전송했습니다. 크기: {len(pcap_data)} 바이트")

        except FileNotFoundError:
            print(f"오류: 캡처 파일 '{pcap_filename}'을 찾을 수 없습니다.")
        except Exception as e:
            print(f"전송 중 오류 발생: {e}")
            
        # 전송 후 파일 삭제
        try:
            os.remove(pcap_filename)
            print(f"임시 파일 '{pcap_filename}'을 삭제했습니다.")
        except OSError as e:
            print(f"파일 삭제 중 오류 발생: {e}")

        # 다음 캡처까지 대기
        time.sleep(CAPTURE_INTERVAL_SECONDS)
        
finally:
    print("스크립트를 종료합니다. 소켓을 닫습니다.")
    sock.close()
