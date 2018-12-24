import cv2

def cascade(video):
    #* openCV provided haar cascades
    eye = cv2.CascadeClassifier("eye_cas.xml")
    face = cv2.CascadeClassifier("face_cas.xml")
    # smile = cv2.CascadeClassifier("smile_cas.xml")

    framesPerSecond = 60

    while 1:
        check, frame = video.read()
        #* The function resize resizes the image src down to or up to the specified size
        #? resize(src, dsize[, dst[, fx[, fy[, interpolation]]]]) -> dst
        frame = cv2.resize(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2)))
        # ? cvtColor(src, code[, dst[, dstCn]]) -> dst
        # * use greyscale (single channel) to remove blobs and draw contours
        # * The function converts an input image from one color space to another
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #* returns  rectange properties of the haar cascade
        faces = face.detectMultiScale(gray, 1.5, 3)
        eyes = eye.detectMultiScale(gray, 1.5, 3)
        # smiles = smile.detectMultiScale(gray, 1.5, 3)

        # * drawing the rectanges
        for (x, y, w, h) in faces:
            # ? rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]]) -> img
            #* The function cv::rectangle draws a rectangle outline or a filled rectangle whose two opposite corners
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # for (sx, sy, sw, sh) in smiles:
            #     cv2.rectangle(frame, (sx, sy), (sx+sw, sy+sh), (255, 0, 0), 2)
            # * drawing both eyes
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(frame,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)

        # ? imshow(winname, mat) -> None
        # * display the frame
        cv2.imshow("capture", frame)

        # * waitkey to escape the program
        key = cv2.waitKey(1000//framesPerSecond)
        if key == ord("q"):
            break


#* captures the videofeed from camera
cap = cv2.VideoCapture(1)
cascade(cap)

#* release everything at the end of the operation
cap.release()
cv2.destroyAllWindows()
