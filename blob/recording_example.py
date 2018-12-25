from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2

vs = cv2.VideoCapture(1)
time.sleep(2.0)
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
fps = 30

writer = None

while True:

	ret, frame = vs.read()

	if writer is None:
		h, w = (frame.shape[0], frame.shape[1])
		writer = cv2.VideoWriter("output.avi", fourcc, fps,
                           (w, h), True)

	output = np.zeros((h, w, 3), dtype="uint8")
	output[0:h, 0:w] = frame

	writer.write(output)


	cv2.imshow("Frame", frame)
	cv2.imshow("Output", output)
	key = cv2.waitKey(16) & 0xFF

	if key == ord("q"):
		break

cv2.destroyAllWindows()
vs.release()
writer.release()
