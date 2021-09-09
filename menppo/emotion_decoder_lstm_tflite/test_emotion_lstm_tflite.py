import sys
# sys.path.append('pingpong')
# from pingpong.pingpongthread import PingPongThread
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

actions = ['easy', 'difficult']
seq_length = 30

# model = load_model('models/model.h5')

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="models/test_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

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

    img = cv2.flip(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = holistic.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if result.face_landmarks is not None:

        joint = np.zeros((468, 4))
        for j, lm in enumerate(result.face_landmarks.landmark):
            joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

        # Compute angles between joints
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

        d = np.concatenate([joint.flatten(), angle])

        seq.append(d)

        # mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

        if len(seq) < seq_length:
            continue

        

        # Test model on random input data.
        # input_shape = input_details[0]['shape']
        # input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
        input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)
        input_data = np.array(input_data, dtype=np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        y_pred = interpreter.get_tensor(output_details[0]['index'])

        i_pred = int(np.argmax(y_pred[0]))
        # conf = y_pred[i_pred]

        # if conf < 0.9:
        #     continue

        action = actions[i_pred]
        action_seq.append(action)

        if len(action_seq) < 3:
            continue

        this_action = '?'
        if action_seq[-1] == action_seq[-2] == action_seq[-3]:
            this_action = action

            if last_action != this_action:
                last_action = this_action
            

        cv2.putText(img, f'{this_action.upper()}', org=(int(result.face_landmarks.landmark[0].x * img.shape[1]), int(result.face_landmarks.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)

    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break

