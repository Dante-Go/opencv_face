import cv2
import numpy as np


def is_inside(o, i):
    ox, oy, ow, oh = o
    ix, iy, iw, ih = i
    return ox > ix and oy > iy and ox+ow < ix+iw and oy+oh < iy+ih


def draw_person(image, person):
    x, y, w, h = person
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 255), 2)


if __name__ == '__main__':
    img = cv2.imread('../images/people.jpg')
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    found, w = hog.detectMultiScale(img, winStride=(8, 8), scale=1.05)

    found_filtered = []
    for ri, r in enumerate(found):
        for qi, q in enumerate(found):
            if ri != qi and is_inside(r, q):
                break
            else:
                found_filtered.append(r)

    for person in found_filtered:
        draw_person(img, person)

    height, width = img.shape[:2]
    reSize = cv2.resize(img, (int(width/2), int(height/2)), interpolation=cv2.INTER_CUBIC)
    cv2.imshow("people detection", reSize)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
