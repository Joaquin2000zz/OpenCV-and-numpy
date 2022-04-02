import cv2
from cv2 import VideoCapture
from cv2 import contourArea
import numpy as np

cap = VideoCapture(0)

while True:
    ret, frame = cap.read()
    # ret will return a true value if the frame exists otherwise False

    into_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # changing the color format from BGr to HSV
    # This will be used to create the masks

    RL_limit = np.array([136, 87, 111]) # setting red lower limit
    RU_limit = np.array([180, 255, 255]) # setting red upper limit
    GL_limit = np.array([25, 52, 72]) # setting green lower limit
    GU_limit = np.array([102, 255, 255]) # setting green upper limit
    BL_limit = np.array([98, 80, 2]) # setting the blue lower limit
    BU_limit = np.array([120, 255, 255]) # setting the blue upper limit
        
    r_mask = cv2.inRange(into_hsv, RL_limit, RU_limit)
    g_mask = cv2.inRange(into_hsv, GL_limit, GU_limit)
    b_mask = cv2.inRange(into_hsv, BL_limit, BU_limit)
    # creating the masks using inRange() function
    # this will produce an image where the color of the objects
    # falling in the range will turn white and rest will be black

    kernal = np.ones((5, 5), "uint8")

    r_mask = cv2.dilate(r_mask, kernal)
    g_mask = cv2.dilate(g_mask, kernal)
    b_mask = cv2.dilate(b_mask, kernal)
    # Morphological Transform, Dilation for each color

    red = cv2.bitwise_and(frame, frame, mask=r_mask)
    green = cv2.bitwise_and(frame, frame, mask=g_mask)
    blue = cv2.bitwise_and(frame, frame, mask=b_mask)
    # bitwise and operator between image frame and mask
    # determine to detect only that particular color

    contours, hierarchy = cv2.findContours(r_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 1000):
            x, y, w, h = cv2.boundingRect(contour)
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            cv2.putText(frame, "Red Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
    # this track the red colour inside a red rectangle

    contours, hierarchy = cv2.findContours(g_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 1000):
            x, y, w, h = cv2.boundingRect(contour)
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.putText(frame, "Green Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0))
    # this track the green colour inside a green rectangle

    contours, hierarchy = cv2.findContours(b_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 1000):
            x, y, w, h = cv2.boundingRect(contour)
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            cv2.putText(frame, "Blue Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0))
    # this track the blue colour inside a blue rectangle

    cv2.imshow("Multiple Color Detection in Real-Time", frame)
    #this shows the frames
    if cv2.waitKey(1) == 27:
        break
    # this function will be triggered when the ESC key is pressed
    # and the while loop will terminate and so will the program
cap.release()
 
cv2.destroyAllWindows()