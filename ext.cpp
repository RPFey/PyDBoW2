#include "DBoW2/TemplatedVocabulary.h"
#include "DBoW2/FORB.h"
#include "DBoW2/BowVector.h"
#include "DBoW2/FeatureVector.h"

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

// pybind11 set up python module
PYBIND11_MODULE(PyDBoW2, m) {
  m.doc() = "DBoW2 vocabulary";

  py::class_<OrbVocabulary>(m, "OrbVocabulary")
      .def(py::init<>())
      .def("loadFromTextFile", &OrbVocabulary::loadFromTextFile)
      .def("getBranchingFactor", &OrbVocabulary::getBranchingFactor)
      .def("getDepthLevels",   &OrbVocabulary::getDepthLevels)
      .def("size", &OrbVocabulary::size)
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

            if(info.stride[1] != 1)
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
}
