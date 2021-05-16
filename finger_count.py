import cv2
import mediapipe as mp
import time
import jetson.utils
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def fingerCount(hand_landmarks, width, half_screen):
    # RIGTH HAND

    isFront = False
    # half_screen = width //2
    print(half_screen, hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * width)
    if hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * width < half_screen:
        # print("RIGTH")
        if(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x > hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x):
            # print("Front")
            isFront = True
        else:
            # print("Back")
            isFront = False
        # THUMB
        if(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x > hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x):
            if isFront:
                isThumb = False
            else:
                isThumb = True
        else:
            if isFront:
                isThumb = True
            else:
                isThumb = False
        # INDEX
        if(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y):
            isIndex = True
        else:
            isIndex = False
        # MIDDLE
        if(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y):
            isMiddle = True
        else:
            isMiddle = False
        # RING
        if(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y):
            isRing = True
        else:
            isRing = False
        # PINKY
        if(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y):
            isPinky = True
        else:
            isPinky = False
    elif hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * width> half_screen:
        # print("LEFT")
        if(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x < hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x):
            # print("Front")
            isFront = True
        else:
            # print("Back")
            isFront = False
        if(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x > hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x):
            if isFront:
                print(4)
                isThumb = True
            else:
                isThumb = False
        else:
            if isFront:
                
                isThumb = False
            else:
                print(4)
                isThumb = True
        # INDEX
        if(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y):
            isIndex = True
        else:
            isIndex = False
        # MIDDLE
        if(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y):
            isMiddle = True
        else:
            isMiddle = False
        # RING
        if(hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y):
            isRing = True
        else:
            isRing = False
        # PINKY
        if(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y):
            isPinky = True
        else:
            isPinky = False
    
    if isFront:
        if isThumb & isIndex & isMiddle & isRing & isPinky:
            return 5
        elif isIndex & isMiddle & isRing & isPinky:
            return 4
        elif isIndex & isMiddle & isRing:
            return 3
        elif isIndex & isMiddle :
            return 2
        elif isIndex:
            return 1
        else:
            return 0
    elif not isFront:
        if isThumb & isIndex & isMiddle & isRing & isPinky:
            return 10
        elif isIndex & isMiddle & isRing & isPinky:
            return 9
        elif isIndex & isMiddle & isRing:
            return 8
        elif isIndex & isMiddle :
            return 7
        elif isIndex:
            return 6
        else:
            return 0
        

if __name__ == "__main__":
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    inp = jetson.utils.videoSource("v4l2:///dev/video0", argv=['--input-flip=horizontal'])
    out = jetson.utils.videoOutput("")
    with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        while True:
            in_img = inp.Capture(format='rgb8')
            jetson.utils.cudaDeviceSynchronize()
            # print(in_img.width, in_img.height)
            resized_img = jetson.utils.cudaAllocMapped(width=in_img.width * 0.75, 
                                                height=in_img.height * 0.75, 
                                                format=in_img.format)
            jetson.utils.cudaResize(in_img, resized_img)
            np_img = jetson.utils.cudaToNumpy(resized_img)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            # print(np_img.shape)
            # np_img = np.fliplr(np_img)
            start_time = time.time()
            np_img.flags.writeable = False
            results = hands.process(np_img)
            end_time = time.time()
            print(end_time - start_time)
            # Draw the hand annotations on the image.
            np_img.flags.writeable = True
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x:
                    #     print(4)
                    # else:print(5)
                    fingerCount(hand_landmarks, resized_img.width)
                    mp_drawing.draw_landmarks(np_img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            out_img = jetson.utils.cudaFromNumpy(np_img)
            out.Render(out_img)
    