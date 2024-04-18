import PyDBoW2

db = PyDBoW2.OrbVocabulary()
loaded = db.loadFromTextFile("/home/wen/Projects/ORB_SLAM2/Vocabulary/ORBvoc.txt")

print(db.getBranchingFactor())