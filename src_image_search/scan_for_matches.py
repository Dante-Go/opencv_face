from os.path import join
from os import walk
import numpy as np
import cv2
from sys import argv


if __name__ == '__main__':
    folder = 'anchors/'
    query = cv2.imread(join(folder, 'tattoo_seed.jpg'), 0)
    files = []
    images = []
    descriptors = []
    for (dirpath, dirnames, filenames) in walk(folder):
        files.extend(filenames)
        for f in files:
            if f.endswith('npy') and f != 'tattoo_seed.npy':
                descriptors.append(f)
        print(descriptors)

    sift = cv2.xfeatures2d.SIFT_create()
    query_kp, query_ds = sift.detectAndCompute(query, None)
    index_params = dict(algorithm=0, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    potential_culprits = {}
    print('Initiating picture scan...')
    for d in descriptors:
        print('--------analyzing %s for matches----------' % d)
        matches = flann.knnMatch(query_ds, np.load(join(folder, d)), k=2)
        good = []
        for m,n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
        if len(good) > 10:
            print('%s is a match! (%d)' % (d, len(good)))
        else:
            print('%s is not a match' % d)
        potential_culprits[d] = len(good)

    max_matches = None
    potential_suspect = None
    for culprit, matches in potential_culprits.items():
        if max_matches == None or matches > max_matches:
            max_matches = matches
            potential_suspect = culprit
    print('potential suspect is %s' % potential_suspect.replace('npy', '').upper())



