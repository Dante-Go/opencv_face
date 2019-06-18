import cv2
import numpy as np
from os.path import join


detect = cv2.xfeatures2d.SIFT_create()
extract = cv2.xfeatures2d.SIFT_create()

flann_params = dict(algorithm=1, trees=5)
matcher = cv2.FlannBasedMatcher(flann_params, {})

# 1.a setup BOW
bow_train = cv2.BOWKMeansTrainer(8)
bow_extract = cv2.BOWImgDescriptorExtractor(extract, matcher)

# 1.b add positives and negatives to the bowtrainer, use SIFT DescriptorExtractor
def feature_sift(fn):
    im = cv2.imread(fn, 0)
    return extract.compute(im, detect.detect(im))[1]

basepath = '../images'
images = ['bb.jpg', 'beans.jpg', 'basil.jpg', 'bathory_album.jpg']

bow_train.add(feature_sift(join(basepath, images[0])))
bow_train.add(feature_sift(join(basepath, images[1])))
bow_train.add(feature_sift(join(basepath, images[2])))
bow_train.add(feature_sift(join(basepath, images[3])))

# 1.c kmeans cluster descriptions to vocabulary
voc = bow_train.cluster()
bow_extract.setVocabulary(voc)
print('bow vocab', np.shape(voc), voc)

# 2.a gather svm training data, use BOWImgDescriptorExtractor
def feature_bow(fn):
    im = cv2.imread(fn, 0)
    return bow_extract.compute(im, detect.detect(im))

traindata, trainlabels =[], []

traindata.extend(feature_bow(join(basepath, images[0])))
traindata.extend(feature_bow(join(basepath, images[1])))
traindata.extend(feature_bow(join(basepath, images[2])))
traindata.extend(feature_bow(join(basepath, images[3])))
trainlabels.append(1)
trainlabels.append(-1)
trainlabels.append(-1)
trainlabels.append(-1)
print('svm items',len(traindata),len(traindata[0]))

# 2.b create & train the svm
svm = cv2.ml.SVM_create()
svm.train(np.array(traindata), cv2.ml.ROW_SAMPLE, np.array(trainlabels))

# 2.c predict the remaining 2*2 images, use BOWImgDescriptorExtractor
def predict(fn):
    f = feature_bow(fn)
    p = svm.predict(f)
    print(fn, '\t', p[1][0][0])

sample = feature_bow(join(basepath, 'bb.jpg'))
p = svm.predict(sample)
print(p)

