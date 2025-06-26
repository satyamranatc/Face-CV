import cv2
import mediapipe as mp
import numpy as np

# Load DNN models for age & gender
age_net = cv2.dnn.readNetFromCaffe("age_deploy.prototxt", "age_net.caffemodel")
gender_net = cv2.dnn.readNetFromCaffe("gender_deploy.prototxt", "gender_net.caffemodel")

AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']

# Initialize mediapipe
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=2, color=(165, 205, 165))

# Start camera
cap = cv2.VideoCapture(0)


def get_orientation(landmarks):
    nose = np.array([landmarks[1].x, landmarks[1].y])
    chin = np.array([landmarks[152].x, landmarks[152].y])
    left = np.array([landmarks[234].x, landmarks[234].y])
    right = np.array([landmarks[454].x, landmarks[454].y])

    vert = chin[1] - nose[1]
    hori = right[0] - left[0]

    if vert > 0.07:
        return "Down"
    elif vert < -0.07:
        return "Up"
    elif hori > 0.5:
        return "Right"
    elif hori < -0.5:
        return "Left"
    else:
        return "Center"


def get_expression(landmarks):
    top_lip = np.array([landmarks[13].x, landmarks[13].y])
    bottom_lip = np.array([landmarks[14].x, landmarks[14].y])
    mouth_open = np.linalg.norm(top_lip - bottom_lip) > 0.03

    left_brow = landmarks[65].y
    left_eye = landmarks[159].y
    brow_eye_dist = abs(left_eye - left_brow)

    if mouth_open and brow_eye_dist > 0.03:
        return "Surprised"
    elif mouth_open:
        return "Smiling"
    else:
        return "Neutral"


def get_face_box(frame, landmarks):
    h, w, _ = frame.shape
    xs = [int(l.x * w) for l in landmarks]
    ys = [int(l.y * h) for l in landmarks]
    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
    padding = 20
    return max(0, x1 - padding), max(0, y1 - padding), min(w, x2 + padding), min(h, y2 + padding)


def predict_age_gender(face_img):
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
    gender_net.setInput(blob)
    gender = GENDER_LIST[gender_net.forward().argmax()]
    age_net.setInput(blob)
    age = AGE_LIST[age_net.forward().argmax()]
    return age, gender


while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    img_h, img_w, _ = frame.shape

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

            landmarks = face_landmarks.landmark

            expression = get_expression(landmarks)
            orientation = get_orientation(landmarks)

            x1, y1, x2, y2 = get_face_box(frame, landmarks)
            face_crop = frame[y1:y2, x1:x2]

            age, gender = predict_age_gender(face_crop) if face_crop.size > 0 else ("?", "?")

            # UI Overlay
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            info = f"{gender}, Age: {age}, {expression}, Looking: {orientation}"
            cv2.putText(frame, info, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    cv2.imshow("Real-Time Face Analysis", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
