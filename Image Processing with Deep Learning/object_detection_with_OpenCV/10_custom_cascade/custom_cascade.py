import cv2
import os

# image storage folder
path = "images"

# image size
imgWidth = 180
imgHeight = 180  # pixel

# video capture
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(3, 480)
cap.set(10, 180)  # brightness

global countFolder
def saveDataFunc():
    global countFolder
    countFolder = 0
    while os.path.exists(path + str(countFolder)):
        countFolder += 1
    os.makedirs(path + str(countFolder))
    
saveDataFunc()

count = 0
countSave = 0

while True:
    
    success, img = cap.read()
    
    if success:
        
        img = cv2.resize(img, (imgWidth, imgHeight))
        
        if count % 5 == 0:
            
            cv2.imwrite(path + str(countFolder) + "/" + str(countSave) + ".png", img)
            countSave += 1 
            print(countSave)
        count += 1
        
        cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()

# %% Creating Cascade
import cv2

path = "cascade.xml"
objectName = "airpods"
frameWidth = 280
frameHeight = 360
color = (255, 0, 255)

cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(3, frameHeight)

def empty():pass

# trackbar
cv2.namedWindow("Result")
cv2.resizeWindow("Result", frameWidth, frameHeight + 100)
cv2.createTrackbar("Scale", "Result", 400, 1000, empty)
cv2.createTrackbar("Neighbor", "Result", 4, 50, empty)


# cascade classifier
cascade = cv2.CascadeClassifier(path)

while True:
    # read image
    success, img = cap.read()
    
    if success:
        # convert BGR2GRAY
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # detection parameters
        scaleVal = 1 + (cv2.getTrackbarPos("Scale", "Result") / 1000)
        neighbor = cv2.getTrackbarPos("Neighbor", "Result")
        
        # detection
        rects = cascade.detectMultiScale(gray, scaleVal, neighbor)
        
        for (x, y, w, h) in rects:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
            cv2.putText(img, objectName, (x, y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)
            
        cv2.imshow("Result", img)
    
    if cv2.waitKey(1) & 0xFF == ord("q"): break
