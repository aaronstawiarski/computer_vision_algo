import cv2
import numpy as np
import gradio as gr
import matplotlib.pyplot as plt

# read the video and capture the size of the window
stream = cv2.VideoCapture(0)
# create background subtraction for better image processing
backSub = cv2.createBackgroundSubtractorMOG2()


if not stream.isOpened():
    print("No stream!")
    exit()

while(True):
    ret, frame = stream.read()
    if not ret :
        print("No more stream!")
        break
    # implementing abstraction & contour algorithms
    fg_mask = backSub.apply(frame)
    cont, hier = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
   

    # filter out / apply temp thresholding
    retval, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
    # want to filter out larger objects for small object detection
    max_obj_area = 2000
    min_obj_area = 300
    small_cont = [cnt for cnt in cont if cv2.contourArea(cnt) <= max_obj_area and cv2.contourArea(cnt) >= min_obj_area]
    frame_out = frame.copy()
    for cnt in small_cont :
        x, y, w, h = cv2.boundingRect(cnt)
        frame_out = cv2.rectangle(frame_out,(x, y), (x + w, y + h), (0, 0, 200), 3 )


    height, width = frame.shape[:2]

    center_x = width // 2
    center_y = height // 2

    cv2.line(frame_out, (center_x - 20, center_y),
             (center_x + 20, center_y), (0, 255, 0), 2)

    cv2.line(frame_out, (center_x, center_y - 20),
             (center_x, center_y + 20), (0, 255, 0), 2)

    



    cv2.imshow("Webcam!", frame_out)
    if cv2.waitKey(1) == ord('q'):
        break

stream.release()
cv2.destroyAllWindows()    