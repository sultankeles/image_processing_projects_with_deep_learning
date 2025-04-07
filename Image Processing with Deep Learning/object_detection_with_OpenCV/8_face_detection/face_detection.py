import cv2
import matplotlib.pyplot as plt


# einstein face detection
einstein = cv2.imread("einstein.jpg", 0)
plt.figure(), plt.imshow(einstein, cmap = "gray"), plt.axis("off")

# classifier
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

face_rect = face_cascade.detectMultiScale(einstein)

for (x, y, w, h) in face_rect:
    cv2.rectangle(einstein, (x, y), (x + w, y + h), (255, 255, 255), 5)

plt.figure(), plt.imshow(einstein, cmap = "gray"), plt.axis("off")


# barcelona players face detection
barcelona = cv2.imread("barcelona.jpg", 0)
plt.figure(), plt.imshow(barcelona, cmap = "gray"), plt.axis("off")

face_rect = face_cascade.detectMultiScale(barcelona, minNeighbors = 7)

for (x, y, w, h) in face_rect:
    cv2.rectangle(barcelona, (x, y), (x + w, y + h), (255, 255, 255), 5)

plt.figure(), plt.imshow(barcelona, cmap = "gray"), plt.axis("off")


# face detection with video camera
cap = cv2.VideoCapture(0)

while True:
    
    ret, frame = cap.read()
    
    if ret:
        
        face_rect = face_cascade.detectMultiScale(frame, minNeighbors = 7) # minNeighbors = 3 is default value
        
        for (x, y, w, h) in face_rect:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (130, 0, 130), 5)
        cv2.imshow("Face Detect", frame)
        
    if cv2.waitKey(1) & 0xFF == ord("q"): break

cap.release()
cv2.destroyAllWindows()
