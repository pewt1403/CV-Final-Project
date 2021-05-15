import cv2
import _thread
import time
import json
import jetson.utils
import face_recognition

USING_RPI_CAMERA_MODULE = False

# SEND DATA TO SERVER
def send_http(img, floor):
    print("Sent")

# INIT CAMERA
inp = jetson.utils.videoSource("v4l2:///dev/video0")
out = jetson.utils.videoOutput("")

i = 0
while True:
    in_img = inp.Capture(format='rgb8')
    print(in_img.width, in_img.height)
    resized_img = jetson.utils.cudaAllocMapped(width=in_img.width * 0.5, 
                                         height=in_img.height * 0.5, 
                                         format=in_img.format)
    jetson.utils.cudaResize(in_img, resized_img)
    np_img = jetson.utils.cudaToNumpy(resized_img)
    # np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

    start_time = time.time()
    face_locations = face_recognition.face_locations(np_img)
    location_time = time.time()
    face_encodings = face_recognition.face_encodings(np_img, face_locations)
    encoding_time = time.time()
    print("Find Location time: {} sec".format(location_time - start_time))
    print("Encoding time: {} sec".format(encoding_time - location_time))
    i += 1
    if i == 5:
        break
    # out.Render(np_img)
    

