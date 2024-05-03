import cv2
import math
import numpy as np
import mediapipe as mp
import skfuzzy as fuzz
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL

def adjust_volume(volume_level):
    volume_level = max(0, min(100, volume_level))
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = interface.QueryInterface(IAudioEndpointVolume)
    volume.SetMasterVolumeLevelScalar(volume_level / 100, None)

def fuzzy_volume_control(distance):
    volume_low = fuzz.trimf(np.arange(0, 101, 1), [0, 0, 40])
    volume_medium = fuzz.trimf(np.arange(0, 101, 1), [20, 50, 80])
    volume_high = fuzz.trimf(np.arange(0, 101, 1), [60, 100, 100])
    rule1 = fuzz.interp_membership(np.arange(0, 101, 1), volume_low, distance)
    rule2 = fuzz.interp_membership(np.arange(0, 101, 1), volume_medium, distance)
    rule3 = fuzz.interp_membership(np.arange(0, 101, 1), volume_high, distance)
    active_rule1 = np.fmin(rule1, volume_low)
    active_rule2 = np.fmin(rule2, volume_medium)
    active_rule3 = np.fmin(rule3, volume_high)
    aggregated = np.fmax(active_rule1, np.fmax(active_rule2, active_rule3))
    if np.sum(aggregated) == 0:
        return 50
    volume_level = fuzz.defuzz(np.arange(0, 101, 1), aggregated, 'centroid')
    volume_level = int(max(0, min(100, volume_level)))
    return volume_level

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

cam = cv2.VideoCapture(0)
while True:
    success, image = cam.read()
    if not success:
        break

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            thumb = hand_landmarks.landmark[4]
            index_finger = hand_landmarks.landmark[8]
            distance = math.sqrt((thumb.x - index_finger.x) ** 2 + (thumb.y - index_finger.y) ** 2)
            distance = distance * 100
            volume_level = fuzzy_volume_control(distance)
            adjust_volume(volume_level)

            h, w, _ = image.shape
            thumb_x = int(thumb.x * w)
            thumb_y = int(thumb.y * h)
            index_x = int(index_finger.x * w)
            index_y = int(index_finger.y * h)

            cv2.circle(image, (thumb_x, thumb_y), 8, (0, 255, 0), -1)
            cv2.circle(image, (index_x, index_y), 8, (0, 0, 255), -1)

    cv2.imshow('Hand Gesture Volume Control', image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
