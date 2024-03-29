import cv2


filename = '../images/vikings.jpg'


def detect(filename):
    face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.namedWindow('Vikings Detected')
    cv2.imshow('Vikings Detected', img)
    cv2.imwrite('../tmp_imgs/vikings.jpg', img)


if __name__ == '__main__':
    detect(filename)
    cv2.waitKey()
