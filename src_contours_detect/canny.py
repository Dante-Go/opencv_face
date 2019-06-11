import cv2
import numpy as np

if __name__ == '__main__':
    img = cv2.imread('images/statue_small.jpg', 0)
    cv2.imwrite('canny2.jpg', cv2.Canny(img, 200, 300))
    cv2.imshow('canny', cv2.imread('canny2.jpg'))
    cv2.waitKey()
    cv2.destroyAllWindows()
