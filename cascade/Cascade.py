import cv2

video = cv2.VideoCapture(1)
eye = cv2.CascadeClassifier("eye_cas.xml")
face = cv2.CascadeClassifier("face_cas.xml")
# smile = cv2.CascadeClassifier("smile_cas.xml")

a:int = 0
while 1:
    a += 1
    print(a)
    check, frame = video.read()
    frame = cv2.resize(frame, (int(frame.shape[1]/2), int(frame.shape[0]/2)))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray, 1.5, 3)
    eyes = eye.detectMultiScale(gray, 1.5, 3)
    # smiles = smile.detectMultiScale(gray, 1.5, 3)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # for (sx, sy, sw, sh) in smiles:
        #     cv2.rectangle(frame, (sx, sy), (sx+sw, sy+sh), (255, 0, 0), 2)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(frame,(ex,ey),(ex+ew,ey+eh),(255,0,0),2)

    cv2.imshow("capture", frame)


    key = cv2.waitKey(1)
    if key == ord("q"):
        break


video.release()
cv2.destroyAllWindows()
