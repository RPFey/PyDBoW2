import numpy as np
from typing import List, Tuple

class Point2f:
    x: float
    y: float

class KeyPoint:
    pt: Point2f
    size: float
    angle: float
    response: float
    octave: int
    class_id: int

class ORBextractor:
    """ ORB feature point extractor"""
    def __init__(self, nfeatures: int, scaleFactor: float, nlevels: int, iniThFAST: int, minThFAST: int) -> None: ...
    
    def extract(self, image: np.ndarray, mask: np.ndarray) -> Tuple[List[KeyPoint], np.ndarray]: 
        """ Extract ORB features from an image."""
        pass

class BRIEFextractor:
    """ BRIEF feature descriptor extractor"""
    def __init__(self, config: str) -> None: ...
    
    def compute(self, image: np.ndarray) -> List[KeyPoint]: 
        """ Compute the BRIEF descriptor from an image and keypoints."""
        pass

class BowVector:
    """ Bag of Words vector """
    def addWeight(self, id: int, weight: float) -> None: ...
    
    def __getitiem__(self, id: int) -> float: ...

class FeatureVector:
    def addFeature(self, id: int) -> None: ...

class OrbVocabulary:

    def loadFromTextFile(self, filename: str) -> None: 
        """ Loads the vocabulary from a text file."""
        pass

    def size(self) -> int: 
        """ Returns the number of words in the vocabulary."""
        pass

    def transform(self, features: np.ndarray, v: BowVector, fv: FeatureVector, nWords: int) -> None: 
        """ Transform features into BoW vector and feature vector."""
        pass
    
    def score(self, v1: BowVector, v2: BowVector) -> float: 
        """ Compute the score between two vectors."""
        pass

class BRIEFVocabulary:

    def __init__(self) -> None: ...

    def size(self) -> int: 
        """ Returns the number of words in the vocabulary."""
        pass

    def transform(self, extractor:BRIEFextractor, img:np.ndarray) -> None: 
        """ Transform features into BoW vector and feature vector."""
        pass
    
    def score(self, v1: BowVector, v2: BowVector) -> float: 
        """ Compute the score between two vectors."""
        pass

class BRIEFLoopDetectorParam:
    height: int
    width: int
    frequency: float
    nss: bool
    alpha: float
    k: int
    dilevels: int
    max_neighbor_ratio: float

    def __init__(self) -> None:
        pass

    def __init__(self, height:int, width:int, frequency: float, 
                        nss: bool, _alpha: float, _k: int, GeometricalCheck, dilevels: int):
        pass

class LoopDetectorParam:
    height: int
    width: int
    frequency: float
    nss: bool
    alpha: float
    k: int
    dilevels: int
    max_neighbor_ratio: float

    def __init__(self) -> None:
        pass

    def __init__(self, height:int, width:int, frequency: float, 
                        nss: bool, _alpha: float, _k: int, GeometricalCheck, dilevels: int):
        pass

class LoopDetectorGeometricalCheck:
    GEOM_EXHAUSTIVE = None
    GEOM_FLANN = None
    GEOM_DI = None
    GEOM_NONE = None

class LoopDetectorDetectionStatus:
    LOOP_DETECTED = None
    CLOSE_MATCHES_ONLY = None

class DetectionResult:
    status: LoopDetectorDetectionStatus
    query: int
    match: int

    def detection():
        pass

class OrbLoopDetector:
    def __init__(self, voc:OrbVocabulary, param:LoopDetectorParam) -> None: ...
    
    def detectLoop(self, keypoints:List[KeyPoint], descriptor:np.ndarray, image_id:int): 
        """ Detect loop closure."""
        pass

class BRIEFLoopDetector:
    def __init__(self, voc:OrbVocabulary, param:LoopDetectorParam) -> None: ...
    
    def detectLoop(self, extractor:BRIEFextractor, image:np.ndarray, image_id:int): 
        """ Detect loop closure."""
        pass
