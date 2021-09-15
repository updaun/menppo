import sys
# sys.path.append('pingpong')
# from pingpong.pingpongthread import PingPongThread
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

actions = ['easy', 'difficult']
seq_length = 30

model = load_model('models/eyebrow_point_model.h5')

# MediaPipe hands model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(
                                min_detection_confidence=0.5,
                                min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)

seq = []
action_seq = []
last_action = None

while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    # img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = holistic.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.face_landmarks is not None:

        joint = np.zeros((468, 4))
        for j, lm in enumerate(result.face_landmarks.landmark):
            joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

        # # Compute angles between joints
        # v1 = joint[[61,146,91,181,84, 17,314,405,321,375, 291,61,185,40,39, 37,0,267,269,270, 409,291], :3] # Parent joint
        # v2 = joint[[146,91,181,84,17, 314,405,321,375,291, 61,185,40,39,37, 0,267,269,270,409, 291,78], :3] # Child joint
        
        # Left eyebrow.
        left_eyebrow = [(276, 283),(283, 282),(282, 295),(295, 285),
                        (300, 293),(293, 334),(334, 296),(296, 336)]

        # Right eyebrow.
        right_eyebrow = [(46, 53),(53, 52),(52, 65),(65, 55),
                        (70, 63),(63, 105),(105, 66),(66, 107)]

        v1 = joint[[x[0] for x in left_eyebrow] + [x[0] for x in right_eyebrow]]
        v2 = joint[[362 for _ in range(len(left_eyebrow))]+[133 for _ in range(len(right_eyebrow))]]
                
        v = v2 - v1 # [20, 3]

        # Normalize v
        # 0 ~ 1사이의 값으로 변환
        v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

        # Get angle using arcos of dot product
        # 각도를 구하는 부분
        angle = np.arccos(np.einsum('nt,nt->n',
            v[[0,1,2,3,4,5,6, 8,9,10,11,12,13,14],:], 
            v[[1,2,3,4,5,6,7, 9,10,11,12,13,14,15],:])) 

        angle = np.degrees(angle) # Convert radian to degree

        d = np.concatenate([joint.flatten(), angle])

        seq.append(d)

        # mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

        if len(seq) < seq_length:
            continue

        input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)

        y_pred = model.predict(input_data).squeeze()

        i_pred = int(np.argmax(y_pred))
        conf = y_pred[i_pred]

        if conf < 0.6:
            continue

        action = actions[i_pred]
        action_seq.append(action)

        if len(action_seq) < 3:
            continue

        this_action = '?'
        if action_seq[-1] == action_seq[-2] == action_seq[-3]:
            this_action = action

            if last_action != this_action:
                last_action = this_action
            

        # cv2.putText(img, f'{this_action.upper()}', org=(int(result.face_landmarks.landmark[0].x * img.shape[1]), int(result.face_landmarks.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        # Grab ear coords
        coords = tuple(np.multiply(
                        np.array(
                            (result.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                                result.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                    , [640,480]).astype(int))
        
        cv2.rectangle(img, 
                        (coords[0], coords[1]+5), 
                        (coords[0]+len(this_action)*20, coords[1]-30), 
                        (245, 117, 16), -1)
        cv2.putText(img, this_action, coords, 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Get status box
        cv2.rectangle(img, (0,0), (250, 60), (245, 117, 16), -1)
        
        # Display Class
        cv2.putText(img, 'CLASS'
                    , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, this_action.split(' ')[0]
                    , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Display Probability
        cv2.putText(img, 'PROB'
                    , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, str(round(y_pred[np.argmax(y_pred)],2))
                    , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


    cv2.imshow('img', img)
    if cv2.waitKey(5) & 0xFF == 27:
        break

