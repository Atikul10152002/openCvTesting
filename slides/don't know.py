import cv2
import numpy as np

# Create an instance of camera 0
cap = cv2.VideoCapture(0)

while True:
    # Get the image from camera 0
    _, image = cap.read()

    # show image under window
    cv2.imshow("Raw Camera Data", image)

    result = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    # Literal values
    HSV_LOW = np.array([0, 20, 0])
    HSV_HIGH = np.array([30, 160, 180])

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

    # result image under window
    cv2.imshow("Result", result)

    # press 'q' key to break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# stop
cv2.destroyAllWindows()
