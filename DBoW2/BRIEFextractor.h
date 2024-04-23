#pragma once

#include "FBrief.h"

#include <DLoopDetector/DLoopDetector.h>
#include <DVision/DVision.h>
#include <DVision/BRIEF.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

/// Generic class to create functors to extract features
template<class TDescriptor>
class FeatureExtractor
{
public:
  virtual ~FeatureExtractor() = default;
  
  /**
   * Extracts features
   * @param im image
   * @param keys keypoints extracted
   * @param descriptors descriptors extracted
   */
  virtual void operator()(
      const cv::Mat& im,
      vector<cv::KeyPoint>& keys,
      vector<TDescriptor>& descriptors) const = 0;
};

/// This functor extracts BRIEF descriptors in the required format
class BriefExtractor: public FeatureExtractor<FBrief::TDescriptor>
{
public:
  virtual ~BriefExtractor() = default;
  
  /** 
   * Extracts features from an image
   * @param im image
   * @param keys keypoints extracted
   * @param descriptors descriptors extracted
   */
  virtual void operator()(
      const cv::Mat& im, 
      vector<cv::KeyPoint>& keys,
      vector<DVision::BRIEF::bitset>& descriptors) const;
  
  /**
   * Creates the brief extractor with the given pattern file
   */
  BriefExtractor(const std::string& pattern_file);

private:
  /// BRIEF descriptor extractor
  DVision::BRIEF m_brief;
};