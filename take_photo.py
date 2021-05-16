import cv2
import _thread
import time
import jetson.utils


inp = jetson.utils.videoSource("v4l2:///dev/video0")
out = jetson.utils.videoOutput("")
out_img = jetson.utils.videoOutput("file:///image.jpg")

while True:
    in_img = inp.Capture(format='rgb8')
    try:
        if keyboard.is_pressed("q"):
            out_img.Render(in_img)
    except:
        pass
