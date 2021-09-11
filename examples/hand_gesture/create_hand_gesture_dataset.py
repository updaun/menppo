import cv2
import mediapipe as mp
import numpy as np
import time, os

# 액션을 4개로 지정
actions = ['go', 'back', 'start', 'finish']
seq_length = 30
# 30초간 촬영을 해서 데이터를 확보하겠다.
secs_for_action = 30

# MediaPipe hands model
# 양손을 찾아주는 솔루션 적용 ( 왼손 좌표는 왼손에, 오른손 좌표는 오른손에 담아줍니다.)
mp_hands = mp.solutions.hands

# 선을 그려주는 모듈
mp_drawing = mp.solutions.drawing_utils

# 클래스 Hands 호출 detector 감지기 
hands = mp_hands.Hands(
    # 최대 하나의 손을 감지
    max_num_hands=1,
    # 감지 정밀도
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# 웹캠으로 사진을 입력 0번 우선순위의 카메라를 사용하겠다.
cap = cv2.VideoCapture(0)

# 현재 시간을 계산 ->  FPS 계산하기 위해서이다.
created_time = int(time.time())

# 데이터셋이라는 폴더 생성
os.makedirs('dataset', exist_ok=True)

# 반복문 : 웹캠이 정상적으로 작동할 때까지 돈다.
while cap.isOpened():
    # 4가지 액션을 반복한다.
    for idx, action in enumerate(actions):
        data = []

        # 이미지 호출
        ret, img = cap.read()

        # 좌우 반전
        img = cv2.flip(img, 1)

        # 글씨 입력
        cv2.putText(img, f'Waiting for collecting {action.upper()} action...', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        # 사진을 보여준다.
        cv2.imshow('img', img)
        # 3초간 보여준다.
        cv2.waitKey(3000)

        start_time = time.time()

        # 30초간 촬영을 하는 부분
        while time.time() - start_time < secs_for_action:
            ret, img = cap.read()

            img = cv2.flip(img, 1)
            # opencv 사진을 찍으면 BGR -> RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # 미디어파이프 통과 부분 (result에 좌표값이 리스트로 들어감)
            result = hands.process(img)
            # 통과된 사진을 RGB -> BGR 바꿔서 추가적인 넘파이 연산을 수행할 것이다.
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # 중요! 만약에 손이 감지된다면?
            if result.multi_hand_landmarks is not None:
                # print(result.multi_hand_landmarks)
                for res in result.multi_hand_landmarks: # 한 손이기때문에 21개의 좌표
                    # 모든 값이 0인 행렬을 만드는데 
                    # x, y, z, v
                    joint = np.zeros((21, 4))
                    for j, lm in enumerate(res.landmark):
                        joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                    # Compute angles between joints (x, y, z 좌표만 계산한다.) 0:x 1:y 2:z
                    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
                    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
                    v = v2 - v1 # [20, 3]
                    # Normalize v
                    # 백터 정규화 -> 화면 사이즈에 구애받지 않고 값을 생성
                    # 0 ~ 1의 값으로 정규환 표현
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                    # Get angle using arcos of dot product
                    # 아크 코사인을 사용하여 각도를 구한다.
                    angle = np.arccos(np.einsum('nt,nt->n',
                        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) # [15,]

                    # 라디안 값을 각도로 변경해준다. (0~6.28(2파이) -> 0 ~ 360)
                    angle = np.degrees(angle) # Convert radian to degree

                    # 각도 계산한 값을 넘파이 배열로 생성
                    angle_label = np.array([angle], dtype=np.float32)
                    angle_label = np.append(angle_label, idx)

                    # 좌표값과 각도값을 concatenate 했다.
                    d = np.concatenate([joint.flatten(), angle_label])

                    data.append(d)
                    # 드로잉 모듈을 사용해서 손의 그림을 그림
                    mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)
            # 그림그린 이미지를 보여준다.
            cv2.imshow('img', img)

            if cv2.waitKey(5) & 0xFF == 27:
                break

        # 감지된 데이터를 
        data = np.array(data)
        print(action, data.shape)
        np.save(os.path.join('dataset', f'raw_{action}_{created_time}'), data)

        # Create sequence data
        full_seq_data = []
        for seq in range(len(data) - seq_length):
            full_seq_data.append(data[seq:seq + seq_length])

        full_seq_data = np.array(full_seq_data)
        print(action, full_seq_data.shape)
        np.save(os.path.join('dataset', f'seq_{action}_{created_time}'), full_seq_data)
    break
