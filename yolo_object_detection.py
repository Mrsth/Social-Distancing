import cv2
import numpy as np
from scipy.spatial import distance as dist

# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# Loading image
# img = cv2.imread("g2.jpg")
# img = cv2.resize(img, (600,400))
# height, width, channels = img.shape

# Loading video
cap = cv2.VideoCapture("v1.mp4")

# Detecting objects
while True:
    _,frame = cap.read()
    frame = cv2.resize(frame,(800,600))
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    circles = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if class_id == 0 and confidence > 0.9:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                circles.append([center_x,center_y])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    print(indexes)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y = circles[i]
            label = str(classes[class_ids[i]])
            color = (255,0,0)
            #cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.circle(frame,(x,y),20,color,2)
            #cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
            D = dist.cdist(circles, circles, metric="euclidean")
            for i in range(0, D.shape[0]):
                for j in range(i + 1, D.shape[1]):
                    # check to see if the distance between any two
                    # centroid pairs is less than the configured number
                    # of pixels
                    if D[i, j] < 50:
                        # update our violation set with the indexes of
                        # the centroid pairs
                        cv2.putText(frame, "Danger", (x,y), font, 1, (0,255,0), 2)
                        cv2.putText(frame,"Please maintain 6 ft distance",(20,40),font,3,(0,0,255),4)


    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    cv2.imshow("Image", frame)

cap.release()
cv2.destroyAllWindows()