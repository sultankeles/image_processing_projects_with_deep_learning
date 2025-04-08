import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import seaborn as sns

col_list = ["frame_number", "identity_number", "left", "top", "width", "height", "score", "class", "visibility"]

data = pd.read_csv("gt.txt", names = col_list)

plt.figure()
sns.countplot(data["class"])

car = data[data["class"] == 3]

video_path = "MOT17-13-SDP.mp4"

cap = cv2.VideoCapture(video_path)

id1 = 29
numberOfImage = np.max(data["frame_number"])
fps = 25
bound_box_list = []

for i in range(numberOfImage):
    
    ret, frame = cap.read()
    
    if ret:
        frame = cv2.resize(frame, dsize = (960, 540))
        
        filter_id1 = np.logical_and(car["frame_number"] == i + 1, car["identity_number"])
        
        if len(car[filter_id1]) != 0:
            
            x = int(car[filter_id1].left.values[0] / 2)
            y = int(car[filter_id1].top.values[0] / 2)
            w = int(car[filter_id1].width.values[0] / 2)
            h = int(car[filter_id1].height.values[0] / 2)
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 180), 2)
            cv2.circle(frame, (int(x + w / 2), int(y + h / 2)), 2, (130, 0, 130), -1)
            
            # frame, x, y, width, height, center_x, center_y
            bound_box_list.append((i, x, y, w, h, int(x + w / 2), int(y + h / 2)))
            
        cv2.putText(frame, "Frame number:" + str(i + 1), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (130, 0, 130), 2)
        cv2.imshow("frame", frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"): break

    else: break

cap.release()
cv2.destroyAllWindows()

df = pd.DataFrame(bound_box_list, columns = ["frame_number", "x", "y", "w", "h", "center_x", "ceter_y"])

df.to_csv("gt_new.txt", index = False)
