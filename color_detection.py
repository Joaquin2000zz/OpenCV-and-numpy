import cv2
import numpy as np


def apply_mask(hsv_frame, frame, L_limit: np.ndarray, U_limit: np.ndarray,
               color: str):
    """
    apply mask given range of colors
    @hsv_frame: frame in hue saturation value format
    @L_limit: lower limit
    @U_limit: upper limit
    @color: color name
    Returns: the modifyed frame
    """
    col_dict = {
        'red': (0, 0, 255),
        'green': (0, 255, 0),
        'blue': (255, 0, 0)
        }
    # creating the masks using inRange() function
    # this will produce an image where the color of the objects
    # falling in the range will turn white and rest will be black
    mask = cv2.inRange(hsv_frame,
                       L_limit, U_limit)
    kernel = np.ones(shape=(5, 5),
                     dtype=np.uint8)
    mask = cv2.dilate(mask,
                      kernel) # Morphological Transform, Dilation for each color
    # bitwise and operator between image frame and mask
    # determine to detect only that particular color
    bitwise = cv2.bitwise_and(frame, frame,
                           mask=mask)
    # this track the red colour inside the color rectangle
    contours, hierarchy = cv2.findContours(mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if (area > 1000):
            x, y, w, h = cv2.boundingRect(contour)
            frame = cv2.rectangle(frame, (x, y),
                                  (x + w, y + h), col_dict[color], 2)

            cv2.putText(frame, color + " Colour", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, col_dict[color])
    return frame

cap = cv2.VideoCapture(0)

while True:
    # ret will return a true value if the frame exists otherwise False
    ret, frame = cap.read()

    # changing the color format from BGR to HSV
    # This will be used to create the masks
    into_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    RL_limit = np.array([136, 87, 111]) # setting red lower limit
    RU_limit = np.array([180, 255, 255]) # setting red upper limit
    GL_limit = np.array([25, 52, 72]) # setting green lower limit
    GU_limit = np.array([102, 255, 255]) # setting green upper limit
    BL_limit = np.array([98, 80, 2]) # setting the blue lower limit
    BU_limit = np.array([120, 255, 255]) # setting the blue upper limit

    frame = apply_mask(into_hsv, frame, RL_limit, RU_limit, 'red')
    frame = apply_mask(into_hsv, frame, GL_limit, GU_limit, 'green')
    frame = apply_mask(into_hsv, frame, BL_limit, BU_limit, 'blue')
    #this shows the frames
    cv2.imshow("Multiple Color Detection in Real-Time", frame)
    # this function will be triggered when the ESC key is pressed
    # and the while loop will terminate and so will the program
    if cv2.waitKey(1) == 27:
        break

cap.release() 
cv2.destroyAllWindows()
