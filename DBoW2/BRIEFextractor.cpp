#include "BRIEFextractor.h"

BriefExtractor::BriefExtractor(const std::string &pattern_file) {
  // The DVision::BRIEF extractor computes a random pattern by default when
  // the object is created.
  // We load the pattern that we used to build the vocabulary, to make
  // the descriptors compatible with the predefined vocabulary
  
  // Loads the pattern
  cv::FileStorage fs(pattern_file.c_str(), cv::FileStorage::READ);
  if (!fs.isOpened()) {
    throw string("Could not open file ") + pattern_file;
  }
  
  vector<int> x1, y1, x2, y2;
  fs["x1"] >> x1;
  fs["x2"] >> x2;
  fs["y1"] >> y1;
  fs["y2"] >> y2;
  
  m_brief.importPairs(x1, y1, x2, y2);
}

void BriefExtractor::operator() (
    const cv::Mat &im, 
    vector<cv::KeyPoint> &keys,
    vector<DVision::BRIEF::bitset> &descriptors) const {
  // Extract FAST keypoints with opencv
  const int fast_th = 20; // corner detector response threshold
  cv::FAST(im, keys, fast_th, true);
  
  // Compute their BRIEF descriptor
  m_brief.compute(im, keys, descriptors);
}