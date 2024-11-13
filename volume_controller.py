import cv2
import pyautogui
import mediapipe as mp  # Corrected import alias

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Convert the BGR image to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the RGB image
        results = hands.process(image_rgb)  # Corrected variable name

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks on the frame
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                # Get the y-coordinates of the index finger tip and thumb tip
                index_finger_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
                thumb_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y

                # Determine the gesture
                if index_finger_y < thumb_y:
                    hand_gesture = 'pointing up'
                elif index_finger_y > thumb_y:
                    hand_gesture = 'pointing down'
                else:
                    hand_gesture = 'other'

                # Perform actions based on the gesture
                if hand_gesture == 'pointing up':
                    pyautogui.press('volumeup')
                    print("Volume Up")
                elif hand_gesture == 'pointing down':
                    pyautogui.press('volumedown')
                    print("Volume Down")

        # Display the resulting frame
        cv2.imshow('Hand Gesture', frame)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
