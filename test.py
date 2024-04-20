import PyDBoW2
import cv2
import numpy as np

img = cv2.imread("/home/wen/Projects/MonoGS/datasets/tum/rgbd_dataset_freiburg1_desk/rgb/1305031473.196069.png")
img2 = cv2.imread("/home/wen/Projects/MonoGS/datasets/tum/rgbd_dataset_freiburg1_desk/rgb/1305031473.095695.png")
mask = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8)

# orb = PyDBoW2.ORBextractor(1000, 1.2, 8, 20, 7)
# kp1, des = orb.extract(img, mask)
# kp2, des2 = orb.extract(img2, mask)

# des = np.array(des)
# des2 = np.array(des2)

orb = cv2.ORB_create()
kp1, des = orb.detectAndCompute(img, None)
kp2, des2 = orb.detectAndCompute(img2, None)

vec, fvec = PyDBoW2.BowVector(), PyDBoW2.FeatureVector()
vec2, fvec2 = PyDBoW2.BowVector(), PyDBoW2.FeatureVector()

voc = PyDBoW2.OrbVocabulary()
voc.loadFromTextFile("/home/wen/Projects/ORB_SLAM2/Vocabulary/ORBvoc.txt")
print(voc.size())

voc.transform(des, vec, fvec, 4)
voc.transform(des2, vec2, fvec2, 4)

s = voc.score(vec, vec2)
print(s)