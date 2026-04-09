import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
from textblob import TextBlob

# Load the trained model
model_dict = pickle.load(open('./model2.p', 'rb'))
model = model_dict['model']

# Initialize webcam
cap = cv2.VideoCapture(0)

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Character labels
labels_dict = {
    0: 'A',  1: 'B',  2: 'C',  3: 'D',  4: 'E',  5: 'F',
    6: 'G',  7: 'H',  8: 'I',  9: 'J', 10: 'K', 11: 'L',
   12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R',
   18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X',
   24: 'Y', 25: 'Z', 26: 'DEL'
}

# Text states
word = ''
sentence = ''
last_seen_time = time.time()

# For delayed gesture capture
current_gesture = ''
gesture_start_time = 0
capture_delay = 2.0  # seconds

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    current_time = time.time()

    if results.multi_hand_landmarks:
        last_seen_time = current_time

        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10
        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        prediction = model.predict([np.asarray(data_aux)])
        predicted_class = int(prediction[0])
        predicted_character = labels_dict.get(predicted_class, '?')

        # Handle DEL gesture (class 26)
        if predicted_character == 'DEL':
            if current_time - gesture_start_time >= capture_delay:
                if word:
                    word = word[:-1]
                    print("Deleted last character. Word is now:", word)
                elif sentence.strip():
                    words = sentence.strip().split()
                    if words:
                        words.pop()
                    sentence = ' '.join(words) + ' ' if words else ''
                    print("Deleted last word. Sentence is now:", sentence)
                gesture_start_time = current_time  # prevent rapid deletions
                current_gesture = 'DEL'
                # continue


        # Normal letter prediction
        if predicted_character == current_gesture:
            if current_time - gesture_start_time >= capture_delay:
                word += predicted_character
                print("Captured Letter:", predicted_character)
                print("Current Word:", word)
                gesture_start_time = current_time
        else:
            current_gesture = predicted_character
            gesture_start_time = current_time

        # Show prediction
        # Draw bounding box and label for ALL characters, including DEL
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)

        cv2.putText(frame, predicted_character, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3,(0, 0, 0), 3, cv2.LINE_AA)

        # Optional: add an extra label for DEL
        if predicted_character == 'DEL':
            cv2.putText(frame, "Deleting...", (x1, y2 + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)


    else:
        # Word complete: hand not seen for 2 seconds
        if current_time - last_seen_time > 2 and word != '':
            b = TextBlob(word.lower())
            corrected = str(b.correct())
            print("TextBlob Corrected:", corrected)
            sentence += corrected + ' '
            print("Sentence:", sentence.strip())
            word = ''
            current_gesture = ''
            gesture_start_time = 0

    # Display sentence
    cv2.putText(frame, sentence.strip(), (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50, 50, 255), 2, cv2.LINE_AA)

    cv2.imshow('frame', frame)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()
