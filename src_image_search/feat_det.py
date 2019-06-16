import cv2
import numpy as np
import sys

def fd(algorithm):
    algorithms = {
        'SIFT': cv2.xfeatures2d.SIFT_create(),
        'SURF': cv2.xfeatures2d.SURF_create(float(sys.argv[3]) if len(sys.argv) == 4 else 4000),
        'ORB': cv2.ORB_create()
    }
    return algorithms[algorithm]


if __name__ == '__main__':
    imgpath = sys.argv[1]
    img = cv2.imread(imgpath)
    alg = sys.argv[2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fd_alg = fd(alg)
    keypoints, descriptior = fd_alg.detectAndCompute(gray, None)
    img = cv2.drawKeypoints(image=img, outImage=img, keypoints=keypoints, flags=4, color=(51, 163, 236))
    cv2.imshow('keypoints', img)
    while(True):
        if cv2.waitKey(int(1000/12)) & 0xff == ord('q'):
            break
    cv2.destroyAllWindows()

