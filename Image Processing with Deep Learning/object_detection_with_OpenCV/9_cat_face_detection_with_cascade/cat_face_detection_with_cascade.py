import cv2
import os

# reading images in folder
files = os.listdir()
print(files)

img_path_list = []
for f in files:
    if f.endswith(".jpg"):
        img_path_list.append(f)
print(img_path_list)

for c in img_path_list:
    print(c)
    image = cv2.imread(c)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    detector = cv2.CascadeClassifier("haarcascade_frontalcatface.xml")
    rects = detector.detectMultiScale(gray, scaleFactor = 1.045, minNeighbors = 2)
    
    for (i, (x, y, w, h)) in enumerate(rects):
        cv2.rectangle(image, (x, y), (x + w, y + h), (130, 0, 130), 2)
        cv2.putText(image, "Cat {}".format(i+1), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (130, 0, 130), 2)
    
    cv2.imshow(c, image)
    if cv2.waitKey(0) & 0xFF == ord("q"): continue
