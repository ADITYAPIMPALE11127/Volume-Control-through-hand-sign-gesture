import cv2
import pyautogui
import mediapipe as mp

# Initialize the default camera (index 0)
cap = cv2.VideoCapture(0)

# Initialize MediaPipe's Hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    min_tracking_confidence=0.5,
    min_detection_confidence=0.5,
    max_num_hands=1
)

mp_drawing = mp.solutions.drawing_utils

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR frame to RGB format
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the hand landmarks using the Hand model
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            # Get y-coordinates of finger tips
            index_finger_y = hand_landmarks.landmark[
                mp_hands.HandLandmark.INDEX_FINGER_TIP
            ].y
            thumb_y = hand_landmarks.landmark[
                mp_hands.HandLandmark.THUMB_TIP
            ].y

            # Determine hand gesture based on finger positions
            if index_finger_y < thumb_y:
                hand_gesture = 'pointing up'
            elif index_finger_y > thumb_y:
                hand_gesture = 'pointing down'
            else:
                hand_gesture = 'other'

            # Perform volume control based on hand gesture
            if hand_gesture == 'pointing up':
                pyautogui.press('volumeup')
            elif hand_gesture == 'pointing down':
                pyautogui.press('volumedown')

    # Display the frame with annotations
    cv2.imshow('HAND Gesture', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
