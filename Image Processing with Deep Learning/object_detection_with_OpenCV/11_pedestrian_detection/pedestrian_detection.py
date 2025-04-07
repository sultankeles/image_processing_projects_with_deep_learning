import cv2
import os

files = os.listdir()
img_path_list = []

for f in files:
    if f.endswith(".jpg"):
        img_path_list.append(f)

print(img_path_list)

# hog identifier
hog = cv2.HOGDescriptor()
# add SVM
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

for imagePath in img_path_list:
    print(imagePath)
    
    image = cv2.imread(imagePath)
    
    (rects, weights) = hog.detectMultiScale(image, padding = (8, 8), scale = 1.05)
    
    for (x, y, w, h) in rects:
        cv2.rectangle(image, (x, y), (x + w, y + h), (130, 0, 130), 2)    
    
    cv2.imshow("Pedestrian", image)
    
    if cv2.waitKey(0) & 0xFF == ord("q"): continue