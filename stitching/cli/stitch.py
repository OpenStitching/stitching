"""
Command line tool for the stitching package
"""

import argparse
import glob

import cv2 as cv
import numpy as np

from stitching import AffineStitcher, Stitcher
from stitching.blender import Blender
from stitching.camera_adjuster import CameraAdjuster
from stitching.camera_estimator import CameraEstimator
from stitching.camera_wave_corrector import WaveCorrector
from stitching.cropper import Cropper
from stitching.exposure_error_compensator import ExposureErrorCompensator
from stitching.feature_detector import FeatureDetector
from stitching.feature_matcher import FeatureMatcher
from stitching.image_handler import ImageHandler
from stitching.seam_finder import SeamFinder
from stitching.subsetter import Subsetter
from stitching.timelapser import Timelapser
from stitching.warper import Warper

parser = argparse.ArgumentParser(prog="stitch.py")
parser.add_argument("img_names", nargs="+", help="Files to stitch", type=str)
parser.add_argument(
    "--affine",
    action="store_true",
    help="Overwrites multiple parameters to optimize the stitching for "
    "scans and images captured by specialized devices.",
)
parser.add_argument(
    "--medium_megapix",
    action="store",
    default=ImageHandler.DEFAULT_MEDIUM_MEGAPIX,
    help="Resolution for image registration step. "
    "The default is %s Mpx" % ImageHandler.DEFAULT_MEDIUM_MEGAPIX,
    type=float,
)
parser.add_argument(
    "--detector",
    action="store",
    default=FeatureDetector.DEFAULT_DETECTOR,
    help="Type of features used for images matching. "
    "The default is '%s'." % FeatureDetector.DEFAULT_DETECTOR,
    choices=FeatureDetector.DETECTOR_CHOICES.keys(),
    type=str,
)
parser.add_argument(
    "--nfeatures",
    action="store",
    default=500,
    help="Number of features to be detected per image. "
    "Only used for the detectors 'orb' and 'sift'. "
    "The default is 500.",
    type=int,
)
parser.add_argument(
    "--matcher_type",
    action="store",
    default=FeatureMatcher.DEFAULT_MATCHER,
    help="Matcher used for pairwise image matching. "
    "The default is '%s'." % FeatureMatcher.DEFAULT_MATCHER,
    choices=FeatureMatcher.MATCHER_CHOICES,
    type=str,
)
parser.add_argument(
    "--range_width",
    action="store",
    default=FeatureMatcher.DEFAULT_RANGE_WIDTH,
    help="uses range_width to limit number of images to match with.",
    type=int,
)
parser.add_argument(
    "--try_use_gpu",
    action="store_true",
    help="Try to use CUDA",
)
parser.add_argument(
    "--match_conf",
    action="store",
    help="Confidence for feature matching step. "
    "The default is 0.3 for ORB and 0.65 for other feature types.",
    type=float,
)
parser.add_argument(
    "--confidence_threshold",
    action="store",
    default=Subsetter.DEFAULT_CONFIDENCE_THRESHOLD,
    help="Threshold for two images are from the same panorama confidence. "
    "The default is '%s'." % Subsetter.DEFAULT_CONFIDENCE_THRESHOLD,
    type=float,
)
parser.add_argument(
    "--matches_graph_dot_file",
    action="store",
    default=Subsetter.DEFAULT_MATCHES_GRAPH_DOT_FILE,
    help="Save matches graph represented in DOT language to <file_name> file.",
    type=str,
)
parser.add_argument(
    "--estimator",
    action="store",
    default=CameraEstimator.DEFAULT_CAMERA_ESTIMATOR,
    help="Type of estimator used for transformation estimation. "
    "The default is '%s'." % CameraEstimator.DEFAULT_CAMERA_ESTIMATOR,
    choices=CameraEstimator.CAMERA_ESTIMATOR_CHOICES.keys(),
    type=str,
)
parser.add_argument(
    "--adjuster",
    action="store",
    default=CameraAdjuster.DEFAULT_CAMERA_ADJUSTER,
    help="Bundle adjustment cost function. "
    "The default is '%s'." % CameraAdjuster.DEFAULT_CAMERA_ADJUSTER,
    choices=CameraAdjuster.CAMERA_ADJUSTER_CHOICES.keys(),
    type=str,
)
parser.add_argument(
    "--refinement_mask",
    action="store",
    default=CameraAdjuster.DEFAULT_REFINEMENT_MASK,
    help="Set refinement mask for bundle adjustment. It looks like 'x_xxx', "
    "where 'x' means refine respective parameter and '_' means don't "
    "refine, and has the following format:<fx><skew><ppx><aspect><ppy>. "
    "The default mask is '%s'. "
    "If bundle adjustment doesn't support estimation of selected "
    "parameter then the respective flag is ignored."
    "" % CameraAdjuster.DEFAULT_REFINEMENT_MASK,
    type=str,
)
parser.add_argument(
    "--wave_correct_kind",
    action="store",
    default=WaveCorrector.DEFAULT_WAVE_CORRECTION,
    help="Perform wave effect correction. "
    "The default is '%s'" % WaveCorrector.DEFAULT_WAVE_CORRECTION,
    choices=WaveCorrector.WAVE_CORRECT_CHOICES.keys(),
    type=str,
)
parser.add_argument(
    "--warper_type",
    action="store",
    default=Warper.DEFAULT_WARP_TYPE,
    help="Warp surface type. The default is '%s'." % Warper.DEFAULT_WARP_TYPE,
    choices=Warper.WARP_TYPE_CHOICES,
    type=str,
)
parser.add_argument(
    "--low_megapix",
    action="store",
    default=ImageHandler.DEFAULT_LOW_MEGAPIX,
    help="Resolution for seam estimation and exposure estimation step. "
    "The default is %s Mpx." % ImageHandler.DEFAULT_LOW_MEGAPIX,
    type=float,
)
parser.add_argument(
    "--crop",
    action="store_true",
    help="Crop black borders around images caused by warping using the "
    "largest interior rectangle. "
    "Default is '%s'." % Cropper.DEFAULT_CROP,
)
parser.add_argument(
    "--no-crop",
    action="store_false",
    help="Don't Crop black borders around images caused by warping using the "
    "largest interior rectangle. "
    "Default is '%s'." % (not Cropper.DEFAULT_CROP),
    dest="crop",
)
parser.set_defaults(crop=Cropper.DEFAULT_CROP)
parser.add_argument(
    "--compensator",
    action="store",
    default=ExposureErrorCompensator.DEFAULT_COMPENSATOR,
    help="Exposure compensation method. "
    "The default is '%s'." % ExposureErrorCompensator.DEFAULT_COMPENSATOR,
    choices=ExposureErrorCompensator.COMPENSATOR_CHOICES.keys(),
    type=str,
)
parser.add_argument(
    "--nr_feeds",
    action="store",
    default=ExposureErrorCompensator.DEFAULT_NR_FEEDS,
    help="Number of exposure compensation feed.",
    type=np.int32,
)
parser.add_argument(
    "--block_size",
    action="store",
    default=ExposureErrorCompensator.DEFAULT_BLOCK_SIZE,
    help="BLock size in pixels used by the exposure compensator. "
    "The default is '%s'." % ExposureErrorCompensator.DEFAULT_BLOCK_SIZE,
    type=np.int32,
)
parser.add_argument(
    "--finder",
    action="store",
    default=SeamFinder.DEFAULT_SEAM_FINDER,
    help="Seam estimation method. "
    "The default is '%s'." % SeamFinder.DEFAULT_SEAM_FINDER,
    choices=SeamFinder.SEAM_FINDER_CHOICES.keys(),
    type=str,
)
parser.add_argument(
    "--final_megapix",
    action="store",
    default=ImageHandler.DEFAULT_FINAL_MEGAPIX,
    help="Resolution for compositing step. Use -1 for original resolution. "
    "The default is %s" % ImageHandler.DEFAULT_FINAL_MEGAPIX,
    type=float,
)
parser.add_argument(
    "--blender_type",
    action="store",
    default=Blender.DEFAULT_BLENDER,
    help="Blending method. The default is '%s'." % Blender.DEFAULT_BLENDER,
    choices=Blender.BLENDER_CHOICES,
    type=str,
)
parser.add_argument(
    "--blend_strength",
    action="store",
    default=Blender.DEFAULT_BLEND_STRENGTH,
    help="Blending strength from [0,100] range. "
    "The default is '%s'." % Blender.DEFAULT_BLEND_STRENGTH,
    type=np.int32,
)
parser.add_argument(
    "--timelapse",
    action="store",
    default=Timelapser.DEFAULT_TIMELAPSE,
    help="Output warped images separately as frames of a time lapse movie, "
    "with 'fixed_' prepended to input file names. "
    "The default is '%s'." % Timelapser.DEFAULT_TIMELAPSE,
    choices=Timelapser.TIMELAPSE_CHOICES,
    type=str,
)
parser.add_argument(
    "--preview",
    action="store_true",
    help="Opens a preview window with the stitched result",
)
parser.add_argument(
    "--output",
    action="store",
    default="result.jpg",
    help="The default is 'result.jpg'",
    type=str,
)

__doc__ += "\n" + parser.format_help()


def main():
    args = parser.parse_args()
    args_dict = vars(args)

    # Extract In- and Output
    img_names = args_dict.pop("img_names")
    if len(img_names) == 1:
        img_names = glob.glob(img_names[0])
    preview = args_dict.pop("preview")
    output = args_dict.pop("output")
    print("stitching " + " ".join(img_names) + " into " + output)

    # Create Stitcher
    affine_mode = args_dict.pop("affine")

    if affine_mode:
        stitcher = AffineStitcher(**args_dict)
    else:
        stitcher = Stitcher(**args_dict)

    panorama = stitcher.stitch(img_names)

    cv.imwrite(output, panorama)

    if preview:
        zoom_x = 600.0 / panorama.shape[1]
        preview = cv.resize(panorama, dsize=None, fx=zoom_x, fy=zoom_x)

        cv.imshow(output, preview)
        cv.waitKey()
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
