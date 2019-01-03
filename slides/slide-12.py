import cv2
import numpy as np

#* Create an instance of camera 0
cap = cv2.VideoCapture(0)

while True:
    #* Get the image from camera 0
    _, image = cap.read()

    #* show image under window
    cv2.imshow("Raw Camera Data", image)

    #* press 'q' key to break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# *  When everything done, release the capture
cv2.destroyAllWindows()
