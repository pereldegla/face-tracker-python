from CentroidTracker import *
import time
import imutils
import cv2
import argparse

# construct the argument parse and parse the argument
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--video", help="video input")
args = vars(parser.parse_args())

if args["video"]:
    video = args["video"]
else:
    video = 0
next_frame = 0
prev_frame = 0

#find path of xml file containing haarcascade file
cascPathface = "haarcascade_frontalface_alt2.xml"
# load the harcaascade in the cascade classifier
faceCascade = cv2.CascadeClassifier(cascPathface)
# initialize our centroid tracker and frame dimensions
ct = CentroidTracker()
(H, W) = (None, None)

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")
vs = cv2.VideoCapture(video)
# loop over the frames from the video stream
while True:
    next_frame = time.time()
    # read the next frame from the video stream and resize it
    ret, frame = vs.read()
    frame = imutils.resize(frame, width=400)
    # if the frame dimensions are None, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = faceCascade.detectMultiScale(gray,
                                              scaleFactor=1.1,
                                              minNeighbors=5,
                                              minSize=(60, 60),
                                              flags=cv2.CASCADE_SCALE_IMAGE)

    rects = []

    # loop over the detections
    for box in detections:
        # draw a bounding box surrounding the object so we can
        # visualize it
        x, y, w, h = box[0], box[1], box[2], box[3]
        rects.append(box.astype("int"))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)
        # update our centroid tracker using the computed set of bounding
        # box rectangles
        objects = ct.update(rects)
        # loop over the tracked objects
        for (objectID, centroid) in objects.items():
            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    fps = 1 / (next_frame - prev_frame)
    prev_frame = next_frame
    # converting the fps into integer
    fps = int(fps)
    # converting the fps to string so that we can display it on frame
    # by using putText function
    fps = str(fps)
    # putting the FPS count on the frame
    cv2.putText(frame, fps, (7, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3, cv2.LINE_AA)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    # do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
