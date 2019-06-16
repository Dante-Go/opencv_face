import os
import sys
import cv2
import numpy as np


def normalize(x, low, high, dtype=None):
    """Normalizes a given array in X to a value between low and high."""
    x = np.array(x)
    minX, maxX = np.min(x), np.max(x)
    # normalize to [0...1].
    x = x - float(minX)
    x = x / float(maxX - minX)
    # scale to [low...high].
    x = x * (high-low)
    x = x + low
    if dtype is None:
        return np.asarray(x)
    return np.asarray(x, dtype=dtype)

def read_images(path, sz=None):
    """Reads the images in a given folder, resizes images on the fly if size is given.
    Args:
        path: Path to a folder with subfolders representing the subjects(persons).
        sz: A tuple with the size Resizes
    Returns:
        A list [x, y]
            x: The images, which is a python list of numpy arrays.
            y: The corresponding labels (the unique number of the subject, person) in a list of python.
    """
    c = 0
    x, y = [], []
    for dirname, dirnames, filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname, subdirname)
            for filename in os.listdir(subject_path):
                try:
                    if filename == '.directory':
                        continue
                    filepath = os.path.join(subject_path, filename)
                    im = cv2.imread(os.path.join(subject_path, filename), cv2.IMREAD_GRAYSCALE)
                    if im is None:
                        print( 'image' + filepath + 'is none')
                    # resize to given size (if given)
                    if sz is not None:
                        im = cv2.resize(im, sz)
                    x.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)
                except IOError as err:
                    print('I/O error:', err)
                except:
                    print('Unexcepted error:', sys.exc_info()[0])
                    raise
            c = c + 1
    return [x, y]


def face_rec():
    names = ['lk', 'xx', 'yy']
    [x, y] = read_images('../tmp_imgs/face_data/')
    y = np.asarray(y, dtype=np.int32)
    model = cv2.face.EigenFaceRecognizer_create()
    model.train(np.asarray(x), np.asarray(y))
    camera = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_default.xml')
    while(True):
        read, img = camera.read()
        faces = face_cascade.detectMultiScale(img, 1.3, 5)
        for (x, y, w, h) in faces:
            img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            roi = gray[x:x+w, y:y+h]
            try:
                roi = cv2.resize(roi, (200, 200), interpolation=cv2.INTER_LINEAR)
                print(roi.shape)
                params = model.predict(roi)
                print('Label: %s, Confidence: %.2f' % (params[0], params[1]))
                cv2.putText(img, names[params[0]], (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
                if (params[0] == 0):
                    cv2.imwrite('face_rec.jpg', img)
            except:
                continue
        cv2.imshow('camera', img)
        if cv2.waitKey(int(1000/12)) & 0xff == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    face_rec()
