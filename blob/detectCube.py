import cv2
import numpy as np
from imutils.video import VideoStream
import hsv_val

sliderEnabled = 1


class openCvPipeline:
    # * default color slider positions
    hueLowStart = 0
    hueHighStart = 255
    saturationLowStart = 0
    saturationHighStart = 255
    valueLowStart = 0
    valueHighStart = 255
    hsvMaxValue = 255

    # * reduced frame rate to avoid lag issues of less powerful computers
    # framesPerSecond = 2 if sliderEnabled else 1
    framesPerSecond = 60

    # * slider names
    hh = 'Hue High'
    hl = 'Hue Low'
    sh = 'Saturation High'
    sl = 'Saturation Low'
    vh = 'Value High'
    vl = 'Value Low'
    br = 'Blur'
    wnd = 'Colorbars'
    kernelSize = 'kernel_size'
    kernelDivision = 'kernel_division'

    def __init__(self):
        # * windows for sliders
        # ? namedWindow(winname[, flags]) -> None
        cv2.namedWindow(self.wnd, cv2.WINDOW_AUTOSIZE)

        # ? (bar name, window name, min , max, argument)
        if sliderEnabled:
            cv2.createTrackbar(self.hl, self.wnd, self.hueLowStart,
                               self.hsvMaxValue, self.nothing)
            cv2.createTrackbar(self.hh, self.wnd,
                               self.hueHighStart, self.hsvMaxValue, self.nothing)
            cv2.createTrackbar(self.sl, self.wnd,
                               self.saturationLowStart, self.hsvMaxValue, self.nothing)
            cv2.createTrackbar(self.sh, self.wnd,
                               self.saturationHighStart, self.hsvMaxValue, self.nothing)
            cv2.createTrackbar(self.vl, self.wnd,
                               self.valueLowStart, self.hsvMaxValue, self.nothing)
            cv2.createTrackbar(self.vh, self.wnd,
                               self.valueHighStart, self.hsvMaxValue, self.nothing)
            cv2.createTrackbar(self.br, self.wnd, 0, 100, self.nothing)

        # * Testing with different values to denoise
        # cv2.createTrackbar(self.kernelSize, self.wnd, 0, 10, self.nothing)
        # cv2.createTrackbar(self.kernelDivision, self.wnd, 1, 25, self.nothing)

    def run(self, video):
        self.capture = video

        # * after 100 errors the program breaks
        errors = 0
        while(self.capture.isOpened()):
            self.ret, self.frame = self.capture.read()
            if self.ret == True:
                self.frame = cv2.flip(self.frame, 180)
                # * resizing the frame to better fit the screen
                self.frame = cv2.resize(self.frame,
                                        (int(self.frame.shape[1]/2),
                                         int(self.frame.shape[0]/2)))


                # * returns hueLow, hueHigh, saturationLow, saturationHigh, valueLow, valueHigh, blur
                self.sliderValues = self.getSliderValues()

                #* Returns the masked image
                self.mask = self.getMask(
                    self.frame, *self.sliderValues)

                #* Returns the contour of the masked image
                self.contours = self.getContours(self.mask)

                #* draws circle on the contour
                self.findPart(self.contours)


                cv2.imshow('mask', self.mask)
                cv2.imshow(self.wnd, self.frame)

                key = cv2.waitKey(1000//self.framesPerSecond)
                if key == ord('s') and sliderEnabled:
                    self.writeHSV(*self.sliderValues)
                if key == ord('q'):
                    self.capture.release()
                    break
            else:
                errors += 1
                if errors > 100:
                    break

    def nothing(self, *a, **k):
        '''
        empty function called by trackbars
        '''
        pass

    def writeHSV(self, hueLow, hueHigh, saturationLow, saturationHigh, valueLow, valueHigh, blur):
        '''
        writes calibrated hsv value of target to text file
        writeHSV(self) -> None
        '''

        # * Appending the final HSV values to the `file`
        with open('hsv_val.py', 'a') as file:
            file.write('#'*10 + '\n')
            file.write('hueLow = ' + str(hueLow) + '\n')
            file.write('hueHigh = ' + str(hueHigh) + '\n')
            file.write('saturationLow = ' + str(saturationLow) + '\n')
            file.write('saturationHigh = ' + str(saturationHigh) + '\n')
            file.write('valueLow = ' + str(valueLow) + '\n')
            file.write('valueHigh = ' + str(valueHigh) + '\n')
            file.write('blur = ' + str(blur) + '\n')
            file.close()

    def getSliderValues(self):
        '''
        returns the slider values
        '''
        # * read trackbar positions for each trackbar. The function returns the current position of the specified trackbar
        # ? getTrackbarPos(trackbarname, winname) -> retval
        hueLow = cv2.getTrackbarPos(
            self.hl, self.wnd) if sliderEnabled else hsv_val.hueLow
        hueHigh = cv2.getTrackbarPos(
            self.hh, self.wnd) if sliderEnabled else hsv_val.hueHigh
        saturationLow = cv2.getTrackbarPos(
            self.sl, self.wnd) if sliderEnabled else hsv_val.saturationLow
        saturationHigh = cv2.getTrackbarPos(
            self.sh, self.wnd) if sliderEnabled else hsv_val.saturationHigh
        valueLow = cv2.getTrackbarPos(
            self.vl, self.wnd) if sliderEnabled else hsv_val.valueLow
        valueHigh = cv2.getTrackbarPos(
            self.vh, self.wnd) if sliderEnabled else hsv_val.valueHigh
        blur = (cv2.getTrackbarPos(self.br, self.wnd) if cv2.getTrackbarPos(
            self.br, self.wnd) % 2 != 0 else cv2.getTrackbarPos(
            self.br, self.wnd) + 1) if sliderEnabled else hsv_val.blur

        return hueLow, hueHigh, saturationLow, saturationHigh, valueLow, valueHigh, blur

    def findPart(self, contours):
        '''
        locates object and its centroid
        findPart(self, contours) -> None
        '''

        '''
        # quick and janky but your milege may vary
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(self.frame, center, radius, (0, 255, 0), 2)
        '''
        if len(contours) != 0:
            contour = max(contours, key=cv2.contourArea)
            # ? contourArea(contour[, oriented]) -> retval
            # * The function computes a contour area. Similarly to moments , the area is computed using the Green. formula. Thus, the returned area and the number of non-zero pixels, if you draw the contour using. \  # drawContours or \#fillPoly , can be different. Also, the function will most certainly give a wrong. results for contours with self-intersections
            self.A = cv2.contourArea(contour)

            # ? moments(array[, binaryImage]) -> retval
            # * The function computes moments, up to the 3rd order, of a vector shape or a rasterized shape. The results are returned in the structure cv: : Moments.
            # * Image Moment is a particular weighted average of image pixel intensities
            # * calculates moments of binary image
            self.M = cv2.moments(contour)
            # * Radius = sqrt(Area * Pi)
            self.Radius = int((self.A/3.14)**(.5))
            # ? change this value if target is smaller/larger
            if self.A > 1000:

                # ? drawContours(image, contours, contourIdx, color[, thickness[, lineType[, hierarchy[, maxLevel[, offset]]]]]) -> image
                cv2.drawContours(self.mask, [contour], -1, (0, 255, 0), 3)
                # * uses the contour's 'moment' to find centroid
                if self.M['m00'] != 0:
                    # * calculate x,y coordinate of center
                    self.circleX = int(self.M['m10']/self.M['m00'])
                    self.circleY = int(self.M['m01']/self.M['m00'])
                    self.center = (self.circleX, self.circleY)
                else:
                    self.center = (0, 0)

                # ? circle(img, center, radius, color[, thickness[, lineType[, shift]]]) -> img
                # * The function cv::circle draws a simple or filled circle with a given center and radius

                # * Centroid center circle
                cv2.circle(self.frame, self.center,
                            10, (159, 159, 255), -1)
                # * Centroid surrounding circle
                cv2.circle(self.frame, self.center,
                            self.Radius, (255, 0, 0), 5)

                print("Object is at ", *self.center)


    def getContours(self, mask):
        '''
        analyzes video feed and finds contours
        getContours(self, frame) -> contours
        '''
        # ? cvtColor(src, code[, dst[, dstCn]]) -> dst
        # * use greyscale (single channel) to remove blobs and draw contours
        # * The function converts an input image from one color space to another
        self.grey = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        # blob removal

        # * Return a new array of given shape and type, filled with ones.
        self.morphkernel = np.ones((5, 5), np.uint8)

        # self.dilatekernel = np.ones((5, 5), np.uint8)
        # self.kernel = np.ones((
        #     cv2.getTrackbarPos(self.kernelSize, self.wnd),
        #     cv2.getTrackbarPos(self.kernelSize, self.wnd)),
        #     np.uint8)/cv2.getTrackbarPos(self.kernelDivision, self.wnd)

        # self.eroded = cv2.erode(self.grey, self.kernel, iterations=1)
        # self.dialated = cv2.dilate(self.grey, self.kernel, iterations=1)

        # ? morphologyEx(src, op, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]]) -> dst
        self.morphed = cv2.morphologyEx(
            self.grey, cv2.MORPH_OPEN, self.morphkernel)
        self.morphed = cv2.morphologyEx(
            self.morphed, cv2.MORPH_CLOSE, self.morphkernel)

        # ? findContours(image, mode, method[, contours[, hierarchy[, offset]]]) -> image, contours, hierarchy
        # * The function retrieves contours from the binary image using the algorithm passed as an argument
        self.contours = cv2.findContours(
            self.morphed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
        return self.contours

    def getMask(self, frame, hueLow, hueHigh, saturationLow, saturationHigh, valueLow, valueHigh, blur):
        '''
        applies mask using hsv trackbar values
        getImage(self, frame) -> res
        '''

        # ? GaussianBlur(src, ksize, sigmaX[, dst[, sigmaY[, borderType]]]) -> dst
        frame = cv2.GaussianBlur(frame, (blur, blur), 0)

        # ? cvtColor(src, code[, dst[, dstCn]]) -> dst
        # * The function converts an input image from one color space to another
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # * define range of mask
        self.HSV_LOW = np.array([hueLow, saturationLow, valueLow])
        self.HSV_HIGH = np.array([hueHigh, saturationHigh, valueHigh])

        # * create a mask with the hsv range
        mask = cv2.inRange(hsv, self.HSV_LOW, self.HSV_HIGH)
        # * cancel out everyting that doesn't belong to the mask
        # * computes bitwise conjunction of the two arrays (dst = src1 & src2)
        mask = cv2.bitwise_and(frame, frame, mask=mask)
        return mask


cv = openCvPipeline()

#* captures the videofeed from camera
camera = cv2.VideoCapture(1) #* Try (0) for Windows
cv.run(camera)

#* release everything at the end of the operation
camera.release()
cv2.destroyAllWindows()
