import numpy as np
import cv2

def detect(frame):

    # convert frame from BGR to Gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # edge detection using Canny
    img_edges = cv2.Canny(gray, 50, 190)

    # convert to black and white image
    _, img_thresh = cv2.threshold(img_edges, 254, 255, cv2.THRESH_BINARY)

    # find contours
    contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # set the minimum & maximum radius of the accepted circle
    min_radius_thresh = 3
    max_radius_thresh = 30

    centers = []
    for c in contours:

        (x, y), radius = cv2.minEnclosingCircle(c)
        radius = int(radius)

        # get only valid circles
        if (radius > min_radius_thresh) and (radius < max_radius_thresh):
            centers.append((x, y))

    return np.array(centers)