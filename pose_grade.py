from pose_compare import compare_poses
import cv2
import mediapipe as mp
import json

# 정답 자세 JSON 로딩
with open("output/exercise_world.json", "r") as file:
    correct_pose = json.load(file)

# 채점할 관절 넘버 리스트
check_nodes = ["23", "25", "27"]  # 왼쪽 골반, 무릎, 발목

# 채점 마진 (10cm)
margin = 0.1

# 미디어파이프 pose 모델을 가져와 실행
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# 기본 카메라 (웹캠)에서 영상을 캡쳐하기 시작
cap = cv2.VideoCapture(0)

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
    if results.pose_world_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

        # 라벨을 붙여서 JSON 형식으로 변환
        landmarks = [
            (i, {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility})
            for i, lm in enumerate(results.pose_world_landmarks.landmark)
        ]
        landmarks_dict = {f"landmark_{i}": lm_data for i, lm_data in landmarks}

    # 결과 프레임 출력
    cv2.imshow("Mediapipe Pose Detection", frame)

    # 채점 수행
    passed = compare_poses(correct_pose, landmarks_dict, margin, 2, *check_nodes)
    print(passed)

    # Q를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 리소스 정리
cap.release()
cv2.destroyAllWindows()
