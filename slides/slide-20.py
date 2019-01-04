import cv2
import numpy as np

# Create an instance of camera 0
cap = cv2.VideoCapture(0)

win = 'Result'
nothing = lambda *args, **kwargs: None

# create window with name win
cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)


# create trackbars
cv2.createTrackbar('Hue Low',  win, 0, 179, nothing)
cv2.createTrackbar('Hue High', win, 100, 179, nothing)
cv2.createTrackbar('Saturation Low',  win, 0, 255, nothing)
cv2.createTrackbar('Saturation High', win, 100, 255, nothing)
cv2.createTrackbar('Value Low',  win, 0, 255, nothing)
cv2.createTrackbar('Value High', win, 100, 255, nothing)
cv2.createTrackbar('Blur', win, 1, 100, nothing)


while True:
    # Get the image from camera 0
    _, image = cap.read()

    image = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2))

    # show image under window
    cv2.imshow("Raw Camera Data", image)

    result = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    # get values from trackbars
    HueLow = cv2.getTrackbarPos('Hue Low',  win)
    HueHigh = cv2.getTrackbarPos('Hue High', win)
    SatLow = cv2.getTrackbarPos('Saturation Low',  win)
    SatHigh = cv2.getTrackbarPos('Saturation High', win)
    ValLow = cv2.getTrackbarPos('Value Low',  win)
    ValHigh = cv2.getTrackbarPos('Value High', win)
    Blur = cv2.getTrackbarPos('Blur', win)
    Blur = Blur if Blur % 2 == 1 else Blur + 1

    # Literal values
    HSV_LOW = np.array([HueLow, SatLow, ValLow])
    HSV_HIGH = np.array([HueHigh, SatHigh, ValHigh])

    # Filter values with mask
    result = cv2.bitwise_and(
        result,
        result,
        mask=cv2.inRange(result, HSV_LOW, HSV_HIGH)
    )

    # Convert result to BGR then to GRAY
    result = cv2.cvtColor(
        result,
        cv2.COLOR_HSV2BGR
    )

    result = cv2.GaussianBlur(
        result, (Blur, Blur), 0
    )

    # create morph kernel
    morphkernel = np.ones((1, 1), np.uint8)
    # removes specs
    result = cv2.morphologyEx(
        result, cv2.MORPH_OPEN, morphkernel
    )
    # removes holes
    result = cv2.morphologyEx(
        result, cv2.MORPH_CLOSE, morphkernel
    )


    # result image under window
    cv2.imshow(win, result)

    # press 'q' key to break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# stop
cv2.destroyAllWindows()
