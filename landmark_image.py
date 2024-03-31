import cv2
import mediapipe as mp
import json
import os
from tkinter import filedialog
from datetime import datetime

# 인식한 이미지와 JSON 파일을 output 디렉토리에 저장
output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 미디어파이프 pose 모델을 가져와 실행
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils


# 파일 선택 다이얼로그
def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        process_image(file_path)


def process_image(file_path):
    # 이미지 로드
    image = cv2.imread(file_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 이미지에서 포즈 감지
    results = pose.process(image_rgb)

    # 포즈가 있으면 랜드마크 그림
    if results.pose_landmarks:
        annotated_image = image.copy()
        mp_drawing.draw_landmarks(
            annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

        # 라벨을 붙여서 JSON 형식으로 변환
        landmarks = [
            (i, {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility})
            for i, lm in enumerate(results.pose_landmarks.landmark)
        ]
        landmarks_dict = {f"landmark_{i}": lm_data for i, lm_data in landmarks}
        landmarks_json = json.dumps(landmarks_dict, indent=4)

        # 현재 날짜와 시간으로 파일명 생성
        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_file_name = os.path.join(output_dir, f"{current_datetime}.json")
        image_file_name = os.path.join(output_dir, f"{current_datetime}.jpg")

        # JSON 데이터를 파일로 저장
        with open(json_file_name, "w") as file:
            file.write(landmarks_json)
            print(f"JSON data saved as {json_file_name}")

        # 이미지에 랜드마크가 그려진 이미지를 저장
        cv2.imwrite(image_file_name, annotated_image)
        print(f"Annotated image saved as {image_file_name}")


# 파일 열기 버튼 추가
open_image()
