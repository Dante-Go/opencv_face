import cv2
import sys
import numpy as np


if __name__ == '__main__':
    img = cv2.imread('../images/varese.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d.SIFT_create()
    keypoints, descriptor = sift.detectAndCompute(gray, None)
    img = cv2.drawKeypoints(image=img, outImage=img, keypoints=keypoints, flags=4, color=(51, 163, 236))
    cv2.imshow('sift_keypoints', img)
    while (True):
        if cv2.waitKey(int(1000/12)) & 0xff == ord('q'):
            break
    cv2.destroyAllWindows()

