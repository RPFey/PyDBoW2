#include "DBoW2/TemplatedVocabulary.h"
#include "DBoW2/FORB.h"
#include "DBoW2/BowVector.h"
#include "DBoW2/FeatureVector.h"
#include "DBoW2/ORBextractor.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;
using namespace DBoW2;

typedef TemplatedVocabulary<FORB::TDescriptor, FORB> OrbVocabulary;

PYBIND11_MAKE_OPAQUE(BowVector);
PYBIND11_MAKE_OPAQUE(FeatureVector);
PYBIND11_MAKE_OPAQUE(std::vector<unsigned int>);
PYBIND11_MAKE_OPAQUE(std::vector<cv::KeyPoint>);

// pybind11 set up python module
PYBIND11_MODULE(_C, m) {
  m.doc() = "DBoW2 vocabulary";

  py::class_<cv::Point2f>(m, "Point2f")
    .def(py::init<>())
    .def_readwrite("x", &cv::Point2f::x)
    .def_readwrite("y", &cv::Point2f::y)
    .def("__repr__", [](const cv::Point2f &p) {
        stringstream ss;
        // set 2 decimal points
        ss << std::fixed << std::setprecision(2) << "Point(" << p.x << ", " << p.y << ")";
        return ss.str();
    });
  
  py::class_<cv::KeyPoint>(m, "KeyPoint")
    .def(py::init<>())
    .def_readwrite("pt", &cv::KeyPoint::pt)
    .def_readwrite("size", &cv::KeyPoint::size)
    .def_readwrite("angle", &cv::KeyPoint::angle)
    .def_readwrite("response", &cv::KeyPoint::response)
    .def_readwrite("octave", &cv::KeyPoint::octave)
    .def_readwrite("class_id", &cv::KeyPoint::class_id)
    .def("__repr__", [](const cv::KeyPoint &kp) {
        stringstream ss;
        // set 2 decimal points
        ss << std::fixed << std::setprecision(2) << "KeyPoint(" << kp.pt.x << ", " << kp.pt.y << ")";
        return ss.str();
    });

  py::class_<cv::Mat>(m, "CvMat", py::buffer_protocol())
    .def(py::init([](py::buffer b) {

        /* Request a buffer descriptor from Python */
        py::buffer_info info = b.request();

        int rows = info.shape[0];
        int cols = info.shape[1];

        if (info.format == py::format_descriptor<unsigned char>::format() && info.ndim == 2){
          return cv::Mat(rows, cols, CV_8U, info.ptr);
        }
        else if (info.format == py::format_descriptor<float>::format() && info.ndim == 2){
          return cv::Mat(rows, cols, CV_32F, info.ptr);
        }
        else if (info.format == py::format_descriptor<unsigned char>::format() && info.ndim == 3 && info.shape[2] == 3) {
          return cv::Mat(rows, cols, CV_8UC3, info.ptr);
        }
        else if (info.format == py::format_descriptor<float>::format() && info.ndim == 3 && info.shape[2] == 3) {
          return cv::Mat(rows, cols, CV_32FC3, info.ptr);
        }
        else
          throw std::runtime_error("Unsupported type of cv::Mat");

    }))
    .def_buffer([](cv::Mat &m) -> py::buffer_info {
        if(m.type() == CV_8U){
          return py::buffer_info(
            m.data,                               /* Pointer to buffer */
            sizeof(unsigned char),                /* Size of one scalar */
            py::format_descriptor<unsigned char>::format(), /* Python struct-style format descriptor */
            2,                                    /* Number of dimensions */
            { m.rows, m.cols },                   /* Buffer dimensions */
            { sizeof(unsigned char) * m.cols,     /* Strides (in bytes) for each index */
              sizeof(unsigned char) }
          );
        }
        else if (m.type() == CV_8UC3){
          return py::buffer_info(
            m.data,                               /* Pointer to buffer */
            sizeof(unsigned char),                /* Size of one scalar */
            py::format_descriptor<unsigned char>::format(), /* Python struct-style format descriptor */
            3,                                    /* Number of dimensions */
            { m.rows, m.cols, 3 },                /* Buffer dimensions */
            { sizeof(unsigned char) * 3 * m.cols, /* Strides (in bytes) for each index */
              sizeof(unsigned char) * 3,
              sizeof(unsigned char) }
          );
        }
        else if (m.type() == CV_32F){
          return py::buffer_info(
            m.data,                               /* Pointer to buffer */
            sizeof(unsigned char),                /* Size of one scalar */
            py::format_descriptor<unsigned char>::format(), /* Python struct-style format descriptor */
            2,                                    /* Number of dimensions */
            { m.rows, m.cols },                   /* Buffer dimensions */
            { sizeof(unsigned char) * m.cols,     /* Strides (in bytes) for each index */
              sizeof(unsigned char) }
          );
        } 
        else if (m.type() == CV_32FC3){
          return py::buffer_info(
            m.data,                               /* Pointer to buffer */
            sizeof(unsigned char),                /* Size of one scalar */
            py::format_descriptor<unsigned char>::format(), /* Python struct-style format descriptor */
            3,                                    /* Number of dimensions */
            { m.rows, m.cols, 3 },                /* Buffer dimensions */
            { sizeof(unsigned char) * 3 * m.cols, /* Strides (in bytes) for each index */
              sizeof(unsigned char) * 3,
              sizeof(unsigned char) }
          );
        }
        else
          throw std::runtime_error("Unsupported type of cv::Mat");
    });

  py::class_<ORB_SLAM2::ORBextractor>(m, "ORBextractor")
    .def(py::init<int, float, int, int, int>())
    .def("extract", [](ORB_SLAM2::ORBextractor& orb, 
                       py::array_t<unsigned char, py::array::c_style | py::array::forcecast> py_image,
                       py::array_t<unsigned char, py::array::c_style | py::array::forcecast> py_mask){

      // convert to cv::Mat
      py::buffer_info info = py_image.request();
      if (info.ndim != 2)
        throw std::runtime_error("Number of dimensions must be two");
      int rows = info.shape[0];
      int cols = info.shape[1];
      if(info.strides[1] != 1)
        throw std::runtime_error("Second dimension must be contiguous");
      cv::Mat image(rows, cols, CV_8U, info.ptr);

      // convert to cv::Mat
      info = py_mask.request();
      if (info.ndim != 2)
        throw std::runtime_error("Number of dimensions must be two");
      rows = info.shape[0];
      cols = info.shape[1];
      if(info.strides[1] != 1)
        throw std::runtime_error("Second dimension must be contiguous");
      cv::Mat mask(rows, cols, CV_8U, info.ptr);

      std::vector<cv::KeyPoint> mvKeys;
      cv::Mat mDescriptors;

      orb(image, mask, mvKeys, mDescriptors);

      return py::make_tuple(mvKeys, mDescriptors);
    }, py::return_value_policy::take_ownership);

  py::class_<OrbVocabulary>(m, "OrbVocabulary")
      .def(py::init<>())
      .def("loadFromTextFile", &OrbVocabulary::loadFromTextFile)
      .def("getBranchingFactor", &OrbVocabulary::getBranchingFactor)
      .def("getDepthLevels",   &OrbVocabulary::getDepthLevels)
      .def("size", &OrbVocabulary::size)
      .def("score", &OrbVocabulary::score)
      .def("transform", [](OrbVocabulary& orb, 
                          py::array_t<unsigned char, py::array::c_style | py::array::forcecast> py_feat, 
                          BowVector &v, FeatureVector &fv, int levelsup){
            // convert to cv::Mat
            std::vector<FORB::TDescriptor> features;

            // check C contiguous of py_feat
            py::buffer_info info = py_feat.request();
            if (info.ndim != 2)
              throw std::runtime_error("Number of dimensions must be two");

            int num_desc = info.shape[0];
            int feat_dim = info.shape[1];
            features.reserve(num_desc);

            if(info.strides[1] != 1)
              throw std::runtime_error("Second dimension must be contiguous");

            unsigned char *data_ptr = static_cast<unsigned char *>(info.ptr);

            for(int i = 0; i < num_desc; i++){
              features.push_back(cv::Mat(1, feat_dim, CV_8U, data_ptr + i*feat_dim));
            }

            orb.transform(features, v, fv, levelsup);
      });


    py::class_<BowVector>(m, "BowVector")
      .def(py::init<>())
      .def("add_weight", &BowVector::addWeight)
      .def("__getitem__", (WordValue& (BowVector::*)(const WordId&)) &BowVector::operator[], py::return_value_policy::reference_internal)
      .def("keys", [](const BowVector& bv){
          std::vector<WordId> keys;
          for(auto it = bv.begin(); it != bv.end(); it++){
            keys.push_back(it->first);
          }
          return keys;
      });

    py::class_<FeatureVector>(m, "FeatureVector")
      .def(py::init<>())
      .def("add_feature", &FeatureVector::addFeature)
      .def("__getitem__", (std::vector<unsigned int>& (FeatureVector::*)(const NodeId&)) &FeatureVector::operator[], py::return_value_policy::reference_internal)
      .def("keys", [](const FeatureVector& fv){
          std::vector<NodeId> keys;
          for(auto it = fv.begin(); it != fv.end(); it++){
            keys.push_back(it->first);
          }
          return keys;
      });

    
    py::bind_vector<std::vector<unsigned int>>(m, "VectorUInt");
    py::bind_vector<std::vector<cv::KeyPoint>>(m, "VectorKeypoint");
}
