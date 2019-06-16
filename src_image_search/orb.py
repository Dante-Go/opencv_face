import numpy as np
import cv2
from matplotlib import pyplot as plt


if __name__ == '__main__':
    img1 = cv2.imread('../images/manowar_logo.png', 0)
    img2 = cv2.imread('../images/manowar_single.jpg', 0)
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:25], img2, flags=2)
    plt.figure(12)
    plt.subplot(1, 2, 1)
    plt.imshow(img3)
    matches2 = bf.knnMatch(des1, des2, k=1)
    img4 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches2, img2, flags=2)
    plt.subplot(1, 2, 2)
    plt.imshow(img4)
    plt.show()

