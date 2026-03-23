"""
Real-time Shadow Hand controller.
Uses your webcam + MediaPipe to drive the MuJoCo Shadow Hand model.

Run with:  mjpython mujoco_hand_control.py
"""

import time
import numpy as np
import cv2
import mujoco
import mujoco.viewer
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH      = "Right_hand.xml"
LANDMARKER_PATH = "hand_landmarker.task"

# ── landmark indices ────────────────────────────────────────────────────────
W  = 0
T  = [1, 2, 3, 4]
FF = [5, 6, 7, 8]
MF = [9, 10, 11, 12]
RF = [13, 14, 15, 16]
LF = [17, 18, 19, 20]

# ── skeleton connections for drawing ───────────────────────────────────────
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

SMOOTH = 0.5


# ── geometry helpers ────────────────────────────────────────────────────────

def pt(landmarks, idx):
    lm = landmarks[idx]
    return np.array([lm.x, lm.y, lm.z])

def angle_at(a, b, c):
    ba = a - b;  bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8
    return float(np.arccos(np.clip(np.dot(ba, bc) / denom, -1, 1)))

def curl(angle):
    return np.clip((np.pi - angle) / np.pi, 0, 1)

def finger_ctrls(landmarks, mcp_idx, pip_idx, dip_idx, tip_idx, prox_range, tendon_range):
    wrist = pt(landmarks, W)
    mcp   = pt(landmarks, mcp_idx)
    pip   = pt(landmarks, pip_idx)
    dip   = pt(landmarks, dip_idx)
    tip   = pt(landmarks, tip_idx)
    c_mcp = curl(angle_at(wrist, mcp, pip))
    c_pip = curl(angle_at(mcp,   pip, dip))
    c_dip = curl(angle_at(pip,   dip, tip))
    return c_mcp * prox_range, (c_pip + c_dip) * (tendon_range / 2.0)

def abduction(landmarks, idx_a, idx_b):
    wrist = pt(landmarks, W)
    a = pt(landmarks, idx_a) - wrist
    b = pt(landmarks, idx_b) - wrist
    ang = float(np.arccos(np.clip(
        np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8), -1, 1)))
    return np.clip(ang * 0.5 - 0.17, -0.349066, 0.349066)


# ── MediaPipe → MuJoCo mapping ──────────────────────────────────────────────

def landmarks_to_ctrl(landmarks):
    ctrl = np.zeros(20)

    ctrl[8],  ctrl[9]  = finger_ctrls(landmarks, *FF, 1.5708, 3.1415)
    ctrl[11], ctrl[12] = finger_ctrls(landmarks, *MF, 1.5708, 3.1415)
    ctrl[14], ctrl[15] = finger_ctrls(landmarks, *RF, 1.5708, 3.1415)
    ctrl[18], ctrl[19] = finger_ctrls(landmarks, *LF, 1.5708, 3.1415)

    ctrl[7]  = abduction(landmarks, FF[0], MF[0])
    ctrl[10] = abduction(landmarks, MF[0], RF[0])
    ctrl[13] = abduction(landmarks, RF[0], LF[0])

    wrist  = pt(landmarks, W)
    t_cmc  = pt(landmarks, T[0])
    t_mcp  = pt(landmarks, T[1])
    t_ip   = pt(landmarks, T[2])
    t_tip  = pt(landmarks, T[3])
    i_mcp  = pt(landmarks, FF[0])

    ctrl[3] = curl(angle_at(wrist, t_cmc, t_mcp)) * 1.22173
    ctrl[5] = curl(angle_at(t_cmc, t_mcp, t_ip))  * 0.698132
    ctrl[6] = curl(angle_at(t_mcp, t_ip,  t_tip)) * 1.5708
    spread  = angle_at(t_cmc, wrist, i_mcp)
    ctrl[2] = np.clip(spread - 0.8, -1.0472, 1.0472)

    return ctrl


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data  = mujoco.MjData(model)

    base_options = mp_python.BaseOptions(model_asset_path=LANDMARKER_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: cannot open camera")
        return

    ctrl_smooth = np.zeros(20)

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.distance  = 0.6
            viewer.cam.elevation = -20
            viewer.cam.azimuth   = 160

            while viewer.is_running():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                h, w  = frame.shape[:2]

                rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result   = landmarker.detect(mp_image)

                if result.hand_landmarks:
                    lms      = result.hand_landmarks[0]
                    new_ctrl = landmarks_to_ctrl(lms)
                    ctrl_smooth = SMOOTH * ctrl_smooth + (1 - SMOOTH) * new_ctrl
                    data.ctrl[:] = ctrl_smooth

                    # draw skeleton on camera frame
                    pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
                    for a, b in CONNECTIONS:
                        cv2.line(frame, pts[a], pts[b], (0, 200, 255), 2)
                    for p in pts:
                        cv2.circle(frame, p, 4, (255, 255, 255), cv2.FILLED)

                cv2.putText(frame, "Hand Control  |  Q = quit",
                            (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (200, 200, 200), 1)
                cv2.imshow("Camera", frame)

                mujoco.mj_step(model, data)
                viewer.sync()

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
