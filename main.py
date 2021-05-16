import cv2
import _thread
import time
import json
import jetson.utils
import face_recognition
import base64
import requests
import numpy as np
import mediapipe as mp

from collections import defaultdict
from utils.yolo.trt_yolo import TensorRT_YOLO
from utils.yolo.classes import get_idx_to_class
from finger_count import fingerCount

USING_RPI_CAMERA_MODULE = False

# SEND DATA TO SERVER
def send_http(img, floor=0):
    ret, buffer = cv2.imencode(".jpg", img)
    img_text = base64.b64encode(buffer)
    r = requests.post(
        'https://59ja20.deta.dev/update_images',
        json={"data":[[buffer, floor]]}
        )
    print("Sent")

# INIT CAMERA
inp = jetson.utils.videoSource("v4l2:///dev/video0")
out = jetson.utils.videoOutput("")

# YOLO INIT
i = 0
tensorrt_yolo = TensorRT_YOLO()
cls_dict = get_idx_to_class()

# MEDIA PIPE INIT
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# LOAD
# pewt_image = face_recognition.load_image_file("recognotion_image/pewt.jpg")
# pewt_face_encoding = face_recognition.face_encodings(pewt_image)[0]
# known_face_encodings = [
#     pewt_face_encoding
# ]
# known_face_names = [
#     "Pewt"
# ]

# ELAVATOR VAR
indiv_dict = defaultdict(lambda : 0)
known_floor = []
Known_floor_name = []
elav_time = 0
elav_interval = 7

recog_now = []
recog_now_name = []
while True:
    # GET FRAME
    in_img = inp.Capture(format='rgb8')
    jetson.utils.cudaDeviceSynchronize()
    # print(in_img.width, in_img.height)
    resized_img = jetson.utils.cudaAllocMapped(width=in_img.width * 0.5, 
                                         height=in_img.height * 0.5, 
                                         format=in_img.format)
    jetson.utils.cudaResize(in_img, resized_img)
    np_img = jetson.utils.cudaToNumpy(resized_img)
    
    # OBJECT DETECTION
    boxes, confs, clss = tensorrt_yolo.detect(np_img)
    this_floor = []
    hand_boxes = []
    hand_boxes_done = []
    to_floor = []
    for i in range(len(clss)):
        if(clss[i] == 1):
            this_floor.append(tuple([boxes[i][1], boxes[i][2], boxes[i][3], boxes[i][0]]))
            
    for i in range(len(this_floor)):
        ext_right = this_floor[i][1] + (this_floor[i][1] - this_floor[i][3])  if this_floor[i][1] + (this_floor[i][1] - this_floor[i][3]) < np_img.shape[1] else np_img.shape[1] - 1
        ext_left =  this_floor[i][3] - (this_floor[i][1] - this_floor[i][3])  if this_floor[i][3] - (this_floor[i][1] - this_floor[i][3])  > 0 else 0
        hand_boxes.append(tuple([this_floor[i][0], ext_right, this_floor[i][2], ext_left]))
        hand_boxes_done.append(False)
        to_floor.append(-1)
        recog_now.append("")
        recog_now_name.append("")
    start_time = time.time()

    # face_locations = face_recognition.face_locations(np_img)
    # print("Location", this_floor)
    location_time = time.time()

    # FACE RECOGNITION
    face_encodings = face_recognition.face_encodings(np_img, this_floor)
    # print(face_encodings)
    face_counter = 0
    name_arr = []
    for face_encoding in face_encodings:
        str_name = "Unknown_{}".format(face_counter)
        matches = face_recognition.compare_faces(known_floor, face_encoding)
        if len(matches) == 0 or True not in matches:
            known_floor.append(face_encoding)
            # indiv_dict[str_name] = [0, 0]
            Known_floor_name.append(str_name)
            recog_now[face_counter] = face_encoding
            recog_now_name[face_counter] = str_name
            face_counter += 1
        else:
            first_match_index = matches.index(True)
            str_name = recog_now_name[first_match_index]
        name_arr.append(str_name)


        # print(matches)
    encoding_time = time.time()
    
    # HAND LANDMARK
    np_img.flags.writeable = False
    results = hands.process(np_img)
    np_img.flags.writeable = True
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        # if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x:
        #     print(4)
        # else:print(5)
        # print(hand_landmarks)
        for i in range(len(hand_boxes)):
            # print("Compare X", hand_boxes[i][3], hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * np_img.shape[1], hand_boxes[i][1])
            # print("Compare Y", hand_boxes[i][0], hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * np_img.shape[0], hand_boxes[i][2])
            if not hand_boxes_done[i] and hand_boxes[i][3] < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * np_img.shape[1] < hand_boxes[i][1] and hand_boxes[i][0] < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * np_img.shape[0] < hand_boxes[i][2]:
                floor = fingerCount(hand_landmarks, resized_img.width, (hand_boxes[i][3] + hand_boxes[i][1]) // 2)
                to_floor[i] = floor
                # print(floor)
                mp_drawing.draw_landmarks(np_img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                hand_boxes_done[i] = True
                break
            
    hand_time = time.time()

    # VISUALIZE
    print(this_floor)
    overlay = np_img.copy()
    output = np_img.copy()
    print(np_img.shape)
    for i in range(len(this_floor)):
        
        cv2.rectangle(overlay, (this_floor[i][3], this_floor[i][0]), (this_floor[i][1],this_floor[i][2]), (0, 255, 0), 0)
        cv2.rectangle(overlay, (hand_boxes[i][3], hand_boxes[i][0]), (hand_boxes[i][1],hand_boxes[i][2]), (0, 0, 255), 0)
        # cv2.rectangle(overlay, (this_floor[i][0], this_floor[i][1]), (this_floor[i][2],this_floor[i][3]), (255, 0, 0), 0)
        # print(to_floor[i])
        cv2.putText(overlay, "Name: {}".format(name_arr[i]), (hand_boxes[i][3], hand_boxes[i][0]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)
        if(to_floor[i] != -1):
            cv2.putText(overlay, "Name: {}, Floor: {}".format(name_arr[i], to_floor[i]), (hand_boxes[i][3], hand_boxes[i][0]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)
        else:
            cv2.putText(overlay, "Name: {}".format(name_arr[i]), (hand_boxes[i][3], hand_boxes[i][0]), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)
    cv2.addWeighted(overlay, 1, output, 0, 0, output)
    add_weight_time = time.time()
    # print("Find Location time: {} sec".format(location_time - start_time))
    print("Encoding time: {} sec".format(encoding_time - location_time))
    print("Hand time: {} sec".format(hand_time - encoding_time))
    print("Hand time: {} sec".format(add_weight_time - hand_time))
    out_img = jetson.utils.cudaFromNumpy(output)
    out.Render(out_img)
    
