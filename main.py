import cv2
import thread
import time
import json
import jetson.utils

# SEND DATA TO SERVER
def send_http(img, floor):
    print("Sent")

# INIT CAMERA
inp = jetson.utils.videoSource("v4l2:///dev/video0")
out = jetson.utils.videoOutput("")

while True:
    

