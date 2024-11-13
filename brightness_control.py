import mediapipe as mp
import cv2
import numpy as np
import screen_brightness_control as sbc
from math import hypot

cap = cv2.VideoCapture(0)  # Corrected VideoCapture
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

while True:
    success, img = cap.read()
    if not success:
        break  # Exit loop if video frame is not captured

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    lmList = []  # Initialize an empty list for landmarks

    if results.multi_hand_landmarks:
        for handlandmark in results.multi_hand_landmarks:
            for id, lm in enumerate(handlandmark.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
            mpDraw.draw_landmarks(img, handlandmark, mpHands.HAND_CONNECTIONS)
            print(lmList)

        # Make sure lmList is not empty
        if lmList:
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[0][1], lmList[0][2]
            cv2.circle(img, (x1, y1), 4, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 4, (255, 0, 0), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

            length = hypot(x2 - x1, y2 - y1)  # Corrected arguments in hypot
            bright = np.interp(length, [15, 220], [0, 100])
            print(bright, length)
            sbc.set_brightness(int(bright))

    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Corrected waitKey syntax
        break

cap.release()
cv2.destroyAllWindows()