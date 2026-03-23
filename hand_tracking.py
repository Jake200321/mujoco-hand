import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = "hand_landmarker.task"

# Landmark indices
WRIST      = 0
THUMB_TIP  = 4;  THUMB_IP   = 3
INDEX_TIP  = 8;  INDEX_PIP  = 6
MIDDLE_TIP = 12; MIDDLE_PIP = 10
RING_TIP   = 16; RING_PIP   = 14
PINKY_TIP  = 20; PINKY_PIP  = 18

FINGER_TIPS  = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

# Hand skeleton connections (pairs of landmark indices)
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),        # thumb
    (0,5),(5,6),(6,7),(7,8),        # index
    (0,9),(9,10),(10,11),(11,12),   # middle
    (0,13),(13,14),(14,15),(15,16), # ring
    (0,17),(17,18),(18,19),(19,20), # pinky
    (5,9),(9,13),(13,17),           # palm
]

def fingers_up(landmarks, is_right):
    up = []
    if is_right:
        up.append(landmarks[THUMB_TIP].x < landmarks[THUMB_IP].x)
    else:
        up.append(landmarks[THUMB_TIP].x > landmarks[THUMB_IP].x)
    for tip, pip in [(INDEX_TIP, INDEX_PIP), (MIDDLE_TIP, MIDDLE_PIP),
                     (RING_TIP, RING_PIP), (PINKY_TIP, PINKY_PIP)]:
        up.append(landmarks[tip].y < landmarks[pip].y)
    return up

def draw_hand(frame, landmarks):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 255), 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (255, 255, 255), cv2.FILLED)

def main():
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open camera.")
        return

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            for i, (hand_landmarks, handedness) in enumerate(
                zip(result.hand_landmarks, result.handedness)
            ):
                label = handedness[0].display_name  # "Left" or "Right"
                is_right = label == "Right"

                draw_hand(frame, hand_landmarks)

                up = fingers_up(hand_landmarks, is_right)

                # Fingertip labels
                for tip_idx, name, extended in zip(FINGER_TIPS, FINGER_NAMES, up):
                    lm = hand_landmarks[tip_idx]
                    tx, ty = int(lm.x * w), int(lm.y * h)
                    color = (0, 255, 0) if extended else (0, 0, 255)
                    cv2.circle(frame, (tx, ty), 8, color, cv2.FILLED)
                    cv2.putText(frame, name, (tx + 10, ty),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

                # Wrist label
                wrist = hand_landmarks[WRIST]
                wx, wy = int(wrist.x * w), int(wrist.y * h)
                cv2.putText(frame, label, (wx - 20, wy + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Finger count
                cv2.putText(frame, f"{label}: {sum(up)} finger(s) up",
                            (10, 30 + 40 * i),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            cv2.putText(frame, "Press Q to quit", (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.imshow("Hand Tracking", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
