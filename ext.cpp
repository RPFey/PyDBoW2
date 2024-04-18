#include "DBoW2/TemplatedVocabulary.h"
#include "DBoW2/FORB.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

typedef DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB> OrbVocabulary;

// pybind11 set up python module
PYBIND11_MODULE(PyDBoW2, m) {
  m.doc() = "DBoW2 vocabulary";
  py::class_<OrbVocabulary>(m, "OrbVocabulary")
      .def(py::init<>())
      .def("loadFromTextFile", &OrbVocabulary::loadFromTextFile)
      .def("getBranchingFactor", &OrbVocabulary::getBranchingFactor)
      .def("getDepthLevels",   &OrbVocabulary::getDepthLevels);
}
