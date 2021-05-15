import cv2
import numpy as np
import jetson.utils
# import face_

# video_capture = cv2.VideoCapture(0)

inp = jetson.utils.videoSource("v4l2:///dev/video0")
out = jetson.utils.videoOutput("")
while True:
    # ret, frame = video_capture.read()
    # small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
    # cv2.imshow("Video", frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
        # break
    image = inp.Capture(format='rgb8')
    out.Render(image)

# video_capture.release()
# cv2.destroyAllWindows()