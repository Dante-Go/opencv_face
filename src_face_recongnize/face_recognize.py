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


if __name__ == '__main__':
    # This is where we write the images, if an output_dir is given in command line:
    out_dir = None
    [x, y] = read_images('../tmp_imgs/face_data/')
    # Convert labels to 32bit integers.
    y = np.asarray(y, dtype=np.int32)
    # Create the Eigenfaces model. We are going to use the default
    # parameters for this simple example, please read the documentation for
    # thresholding:
    model = cv2.face.EigenFaceRecognizer_create()
    # Learn the model. Remember our function returns Python list,
    # so we use np.asarray to turn them into NumPy lists to make
    # the OpenCV wrapper happy:
    model.train(np.asarray(x), np.asarray(y))
    # We now get a prediction from the model! In reality you should
    # always use unseen images for testing your model.
    # model.predict is going to return the predicted label and the associated confidence.
    [p_label, p_confidence] = model.predict(np.asarray(x[0]))
    print('Predicted label = %d (confidence = %.2f)' % (p_label, p_confidence))
    cv2.waitKey()




