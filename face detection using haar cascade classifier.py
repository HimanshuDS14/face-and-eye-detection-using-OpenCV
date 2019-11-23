import cv2
import datetime

face_classifier = cv2.CascadeClassifier(r"C:\Users\dell\AppData\Local\Programs\Python\Python37\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml")
eye_classifier  = cv2.CascadeClassifier(r"C:\Users\dell\AppData\Local\Programs\Python\Python37\Lib\site-packages\cv2\data\haarcascade_eye.xml")

cap = cv2.VideoCapture(0)

while cap.isOpened():

    ret , frame = cap.read()

    gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)

    datet = str(datetime.datetime.now())




    faces = face_classifier.detectMultiScale(gray , 1.1 , 5)
    frame = cv2.putText(frame , datet , (10,50) , cv2.FONT_HERSHEY_COMPLEX , 1,(0,0,255) , 2)

    for (x,y,h,w) in faces:
        cv2.rectangle(frame , (x,y) , (x+w , y+h) , (0,255,0) , 3)

        roi_gray = gray[y:y+h , x:x+w]
        roi_color = frame[y:y+h , x:x+w]

        eyes = eye_classifier.detectMultiScale(roi_gray)

        for(ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color ,(ex,ey) , (ex+ew , ey+eh) , (255,0,0) , 3)



    cv2.imshow("image" , frame)

    if cv2.waitKey(1)==13:
        break

cap.release()
cv2.destroyAllWindows()