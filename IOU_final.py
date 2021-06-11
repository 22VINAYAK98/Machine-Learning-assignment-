# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 16:20:42 2021

@author: ASUS
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 14:50:04 2021

@author: ASUS
"""

import cv2
import numpy as np

# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
GDT_values = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
    
with open("TopDownHuman-Ground_Truth.txt", "r") as f:
    GDT_values =[line.strip() for line in f.readlines()]
    
print(GDT_values[1])
my_points = []


####################################################


def get_iou(a, b, epsilon=1e-5):
    """ Given two boxes `a` and `b` defined as a list of four numbers:
            [x1,y1,x2,y2]
        where:
            x1,y1 represent the upper left corner
            x2,y2 represent the lower right corner
        It returns the Intersect of Union score for these two boxes.

    Args:
        a:          (list of 4 numbers) [x1,y1,x2,y2]
        b:          (list of 4 numbers) [x1,y1,x2,y2]
        epsilon:    (float) Small value to prevent division by zero

    Returns:
        (float) The Intersect of Union score.
    """
    # COORDINATES OF THE INTERSECTION BOX
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    # AREA OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)
    # handle case where there is NO overlap
    if (width<0) or (height <0):
        return 0.0
    area_overlap = width * height

    # COMBINED AREA
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined+epsilon)
    return iou


####################################################################

for i in range(len(GDT_values)) :
    x1 = GDT_values[i].split(' ')[1]
    y1 = GDT_values[i].split(' ')[2]
    w1 = GDT_values[i].split(' ')[3]
    h1 = GDT_values[i].split(' ')[4]
    my_points.append([float(x1),float(y1),float(w1),float(h1)])
    
#print(my_points[1])
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# Loading image
img = cv2.imread("TopDownHumanDetection_4032x3024.jpg")
img = cv2.resize(img, None, fx=0.4, fy=0.4)
height, width, channels = img.shape

# Detecting objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (800,800), (0, 0, 0), True, crop=False)

net.setInput(blob)
outs = net.forward(output_layers)

# Showing informations on the screen
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)
            
            

iou_list = []            
font = cv2.FONT_HERSHEY_PLAIN        
count1 = 0
labeled_ground = []
for i in range(len(my_points)):
    
    c_x2, c_y2, w2, h2 = my_points[i]
    c_x2 = int(c_x2*width)
    c_y2 = int(c_y2*height)
    w2 = int(h2*width)
    h2 = int(h2*height)
    
    
    x2 = int(c_x2 - w2 / 2)
    y2 = int(c_y2 - h2 / 2)
    count1 +=1
    cv2.rectangle(img, (x2, y2), (x2 + w2, y2 + h2), (255,0,255), 1)
    labeled_ground.append([x2, y2, x2 + w2, y2 + h2])
    cv2.putText(img, str(count1) , (x2+10, y2 + 30), font, 1, (255,0,255), 1)            
            
            
            
            

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
print(indexes)
count = 0
predicted = []

for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        print(x,y,w,h)
        label = str(classes[class_ids[i]])
        print(label)
        confi = str(round(confidences[i], 2)*100)
        count +=1
        imge_no = str(count)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0,0,255), 1)
        predicted.append([x,y,x+w,y+h])
        max_iou = 0
        for j in range(len(labeled_ground)):
            
            iou_1 = get_iou([x,y,x+w,y+h],labeled_ground[j])
            print(iou_1)
            if iou_1 > max_iou:
                max_iou = iou_1
            
        iou_ = str(int(max_iou*100 ))
        cv2.putText(img, imge_no+")" +label+":"+ confi+ " %" + " , IOU : " + iou_ , (x+10, y + 30), font, 1, (0,0,255), 2)

# for i in range(len(predicted)):
#     for j in range(len(labeled_ground)):
#          iou_1 = get_iou(predicted[i],labeled_ground[j])
#          if iou_1 >0:
#              iou_list.append(iou_1)
                

print(count)

cv2.imshow("Image", img) 
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('Prediction+ confidence+IOU.jpg ', img)