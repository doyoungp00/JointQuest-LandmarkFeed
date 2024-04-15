import math
import cv2
import mediapipe as mp

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

    # 운동 점수화
    score = 0
    if all(
        key in landmarks_dict
        for key in ["landmark_24", "landmark_25", "landmark_26", "landmark_27"]
    ):
        right_pelvis = landmarks_dict["landmark_24"]
        right_knee = landmarks_dict["landmark_26"]
        left_knee = landmarks_dict["landmark_25"]
        left_ankle = landmarks_dict["landmark_27"]

        # 무릎과 발목의 x좌표가 비슷한지 확인 (지면과 수직)
        if abs(left_knee["x"] - left_ankle["x"]) < margin:
            # 왼쪽 무릎은 오른쪽 골반과 오른쪽 무릎의 y값 사이를 움직임
            # 슬라이더처럼 취급
            if right_pelvis["y"] != right_knee["y"]:
                score = (left_knee["y"] - right_knee["y"]) / (
                    right_pelvis["y"] - right_knee["y"]
                )
                score = max(0.02, min(score, 1))  # 0과 1 사이로 clamping
                score = math.log(score, 2) / 6 + 1
                score = max(0, min(score, 1))

            print(f"Score: {score:.3f}")

    # Q를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 리소스 정리
cap.release()
cv2.destroyAllWindows()
