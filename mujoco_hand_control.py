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


SMOOTH = 0.05


# ── geometry helpers ────────────────────────────────────────────────────────

def pt(landmarks, idx):
    lm = landmarks[idx]
    return np.array([lm.x, lm.y, lm.z])

def angle_at(a, b, c):
    ba = a - b;  bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8
    return float(np.arccos(np.clip(np.dot(ba, bc) / denom, -1, 1)))

def curl(angle):
    # 180° = 0 (straight), 90° = 1 (fully curled)
    return np.clip((np.pi - angle) / (np.pi / 2), 0, 1)

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

    # ── wrist rotation ────────────────────────────────────────────────────
    # Use the vector from wrist → middle MCP as the hand's orientation axis.
    mid_mcp  = pt(landmarks, MF[0])
    hand_vec = mid_mcp - wrist
    xy_len   = np.sqrt(hand_vec[0]**2 + hand_vec[1]**2) + 1e-8

    # WRJ2 (ctrl[0]) – radial/ulnar deviation: hand tilts left/right (x-axis in image)
    lateral = hand_vec[0] / xy_len
    ctrl[0] = np.clip(lateral * 0.5, -0.523599, 0.174533)

    # WRJ1 (ctrl[1]) – flex/extension: hand tilts up/down (y-axis in image)
    # negate because image y increases downward
    vertical = -hand_vec[1] / xy_len
    ctrl[1]  = np.clip(vertical * 0.6, -0.698132, 0.488692)

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
                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result   = landmarker.detect(mp_image)

                if result.hand_landmarks:
                    lms      = result.hand_landmarks[0]
                    new_ctrl = landmarks_to_ctrl(lms)
                    ctrl_smooth = SMOOTH * ctrl_smooth + (1 - SMOOTH) * new_ctrl
                    data.ctrl[:] = ctrl_smooth

                mujoco.mj_step(model, data)
                viewer.sync()

    cap.release()


if __name__ == "__main__":
    main()
