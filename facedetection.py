import cv2
from imutils.video import VideoStream
import imutils
#find path of xml file containing haarcascade file
cascPathface = "haarcascade_frontalface_alt2.xml"
# load the harcaascade in the cascade classifier
faceCascade = cv2.CascadeClassifier(cascPathface)
print("Streaming started")
vs = VideoStream(src=0).start()

# loop over frames from the video file stream
while True:
    # grab the frame from the threaded video stream
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    boxes = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
    for box in boxes:
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
vs.stop()
