import cv2
import mediapipe as mp
import json
import tkinter as tk

# 미디어파이프 pose 모델을 가져와 실행
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# 기본 카메라 (웹캠)에서 영상을 캡쳐하기 시작
cap = cv2.VideoCapture(0)

# JSON 창 생성
root = tk.Tk()
root.title("Landmark JSON")
json_text = tk.Text(root, height=20, width=40)
json_text.pack(fill=tk.BOTH, expand=True)

# JSON 창의 자동 크기 조절 금지
root.pack_propagate(False)

# 웹캠에서 넘어오는 각 프레임을 분석
while cap.isOpened():
    ret, frame = cap.read()  # 스트림이 끝나면 ret == False 되어 종료
    if not ret:
        break

    # 프레임을 BGR (opencv 기본 형식) -> RGB (미디어파이프 형식)로 변환
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # 프레임에서 포즈 감지
    results = pose.process(image)

    # 포즈가 있으면 랜드마크 그림
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

        # 라벨을 붙여서 JSON 형식으로 변환
        landmarks = [
            (i, {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility})
            for i, lm in enumerate(results.pose_landmarks.landmark)
        ]
        landmarks_dict = {f"landmark_{i}": lm_data for i, lm_data in landmarks}
        landmarks_json = json.dumps(landmarks_dict, indent=4)

        # 텍스트 상자 업데이트
        json_text.delete("1.0", tk.END)  # Clear previous text
        json_text.insert(tk.END, landmarks_json)

    # 결과 프레임 출력
    cv2.imshow("Mediapipe Pose Detection", frame)

    # 루프마다 tkinter 창 업데이트
    root.update()

    # Q를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 리소스 정리
cap.release()
cv2.destroyAllWindows()
root.mainloop()
