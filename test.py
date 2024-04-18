import PyDBoW2
import cv2
import numpy as np

img = cv2.imread("/home/wen/Projects/MonoGS/datasets/tum/rgbd_dataset_freiburg1_desk/rgb/1305031473.196069.png")
mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8)

orb = cv2.ORB_create()
kpts, des = orb.detectAndCompute(img, mask)

vec = PyDBoW2.BowVector()
fvec = PyDBoW2.FeatureVector()

voc = PyDBoW2.OrbVocabulary()
voc.loadFromTextFile("/home/wen/Projects/ORB_SLAM2/Vocabulary/ORBvoc.txt")
print(voc.size())

voc.transform(des, vec, fvec, 4)
for k in vec.keys():
    print(k, vec[k])