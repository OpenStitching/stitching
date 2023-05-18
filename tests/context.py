import os
import sys

import cv2 as cv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stitching import AffineStitcher, Stitcher  # noqa: F401, E402
from stitching.blender import Blender  # noqa: F401, E402
from stitching.camera_adjuster import CameraAdjuster  # noqa: F401, E402
from stitching.camera_estimator import CameraEstimator  # noqa: F401, E402
from stitching.camera_wave_corrector import WaveCorrector  # noqa: F401, E402
from stitching.cli.stitch import create_parser, main  # noqa: F401, E402
from stitching.cropper import Cropper  # noqa: F401, E402
from stitching.exposure_error_compensator import (  # noqa: F401, E402
    ExposureErrorCompensator,
)
from stitching.feature_detector import FeatureDetector  # noqa: F401, E402
from stitching.feature_matcher import FeatureMatcher  # noqa: F401, E402
from stitching.image_handler import ImageHandler  # noqa: F401, E402
from stitching.megapix_scaler import (  # noqa: F401, E402
    MegapixDownscaler,
    MegapixScaler,
)
from stitching.seam_finder import SeamFinder  # noqa: F401, E402
from stitching.stitching_error import StitchingWarning, StitchingError  # noqa: F401, E402
from stitching.subsetter import Subsetter  # noqa: F401, E402
from stitching.timelapser import Timelapser  # noqa: F401, E402
from stitching.warper import Warper  # noqa: F401, E402

TEST_DIR = os.path.abspath(os.path.dirname(__file__))
IN_DIR = os.path.join(TEST_DIR, "testdata")
OUT_DIR = os.path.join(TEST_DIR, "results")


def load_testimg(img_name):
    return cv.imread(testinput(img_name))


def testinput(file):
    return os.path.join(IN_DIR, file)


def write_testresult(img_name, img):
    cv.imwrite(testoutput(img_name), img)


def testoutput(file):
    return os.path.join(OUT_DIR, file)
