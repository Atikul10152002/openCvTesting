import cv2
import numpy as np

# Create an instance of camera 0
cap = cv2.VideoCapture(0)

win = 'Input'
cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
nothing = lambda *args, **kwargs: None

hl = cv2.createTrackbar('Hue Low',  win, 0, 179, nothing)
hh = cv2.createTrackbar('Hue High', win, 0, 179, nothing)
sl = cv2.createTrackbar('Saturation Low',  win, 0, 255, nothing)
sh = cv2.createTrackbar('Saturation High', win, 0, 255, nothing)
vl = cv2.createTrackbar('Value Low',  win, 0, 255, nothing)
vh = cv2.createTrackbar('Value High', win, 0, 255, nothing)


while True:
    # Get the image from camera 0
    _, image = cap.read()

    # show image under window
    cv2.imshow("Raw Camera Data", image)

    result = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    HueLow = cv2.getTrackbarPos('Hue Low',  win)
    HueHigh = cv2.getTrackbarPos('Hue High', win)
    SatLow = cv2.getTrackbarPos('Saturation Low',  win)
    SatHigh = cv2.getTrackbarPos('Saturation High', win)
    ValLow = cv2.getTrackbarPos('Value Low',  win)
    ValHigh = cv2.getTrackbarPos('Value High', win)

    # Literal values
    HSV_LOW = np.array([HueLow, SatLow, ValLow])
    HSV_HIGH = np.array([HueHigh, SatHigh, ValHigh])

    # Filter values with mask
    result = cv2.bitwise_and(
        result,
        result,
        mask=cv2.inRange(result, HSV_LOW, HSV_HIGH)
    )

    result = cv2.cvtColor(
        result,
        cv2.COLOR_HSV2BGR
    )
    result = cv2.cvtColor(
        result,
        cv2.COLOR_BGR2GRAY
    )

    # result image under window
    cv2.imshow("Result", result)

    # press 'q' key to break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# stop
cv2.destroyAllWindows()
