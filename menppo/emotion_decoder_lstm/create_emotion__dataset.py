import cv2
import mediapipe as mp
import numpy as np
import time, os

actions = ['easy', 'difficult']
seq_length = 30
secs_for_action = 30

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

created_time = int(time.time())
os.makedirs('dataset', exist_ok=True)

while cap.isOpened():
    for idx, action in enumerate(actions):
        data = []

        ret, img = cap.read()

        # img = cv2.flip(img, 1)

        cv2.putText(img, f'Waiting for collecting {action.upper()} action...', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        cv2.imshow('img', img)
        cv2.waitKey(3000)

        start_time = time.time()

        while time.time() - start_time < secs_for_action:
            ret, img = cap.read()

            # img = cv2.flip(img, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = holistic.process(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # face = result.face_landmarks.landmark
            # face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face]).flatten())

            if result.face_landmarks is not None:
                # for res in result.face_landmarks:
                joint = np.zeros((468, 4))
                for j, lm in enumerate(result.face_landmarks.landmark):
                    joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                # Compute angles between joints
                # v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :3] # Parent joint
                # v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :3] # Child joint
                # 78,95,88,178,87,14,317,402,318,324,308,78,
                v1 = joint[[61,146,91,181,84, 17,314,405,321,375, 291,61,185,40,39, 37,0,267,269,270, 409,291], :3] # Parent joint
                v2 = joint[[146,91,181,84,17, 314,405,321,375,291, 61,185,40,39,37, 0,267,269,270,409, 291,78], :3] # Child joint

                v = v2 - v1 # [20, 3]
                # Normalize v
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                # Get angle using arcos of dot product
                angle = np.arccos(np.einsum('nt,nt->n',
                    v[[0,1,2,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:], 
                    v[[1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21],:])) 

                angle = np.degrees(angle) # Convert radian to degree

                angle_label = np.array([angle], dtype=np.float32)
                angle_label = np.append(angle_label, idx)

                d = np.concatenate([joint.flatten(), angle_label])

                data.append(d)

                mp_drawing.draw_landmarks(img, result.face_landmarks, mp_holistic.FACE_CONNECTIONS)

            cv2.imshow('img', img)
            if cv2.waitKey(5) & 0xFF == 27:
                break

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
