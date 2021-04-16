import cv2

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#         For Face-Detection In An Image

img = cv2.imread('face-image.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(
    gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30)
)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('Faces', img)

cv2.waitKey()

#         For Face-Detection In A Videos

video = cv2.VideoCapture('tkss.mp4')

while video.isOpened():
    _, vid = video.read()
    gray = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(30,30))
    for(x,y,w,h) in faces:
        cv2.rectangle(vid, (x,y), (x+w, y+h), (0,255,0), 2)

    cv2.imshow('faceDetector', vid)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;

video.release()
