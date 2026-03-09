import warnings
from types import SimpleNamespace
import numpy as np
import cv2 as cv
import json
import os
from cv2.detail import CameraParams
from pathlib import Path

from .blender import Blender
from .camera_adjuster import CameraAdjuster
from .camera_estimator import CameraEstimator
from .camera_wave_corrector import WaveCorrector
from .cropper import Cropper
from .exposure_error_compensator import ExposureErrorCompensator
from .feature_detector import FeatureDetector
from .feature_matcher import FeatureMatcher
from .images import Images
from .seam_finder import SeamFinder
from .stitching_error import StitchingError, StitchingWarning
from .subsetter import Subsetter
from .timelapser import Timelapser
from .verbose import verbose_stitching
from .warper import Warper

def convert(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError("Object of type '%s' is not JSON serializable" % type(obj).__name__)

class Stitcher:
    DEFAULT_SETTINGS = {
        "medium_megapix": Images.Resolution.MEDIUM.value,
        "detector": FeatureDetector.DEFAULT_DETECTOR,
        "nfeatures": 500,
        "matcher_type": FeatureMatcher.DEFAULT_MATCHER,
        "range_width": FeatureMatcher.DEFAULT_RANGE_WIDTH,
        "try_use_gpu": False,
        "match_conf": None,
        "calibrate": False,
        "calibration_file": None,
        "megapixels": '16',
        "confidence_threshold": Subsetter.DEFAULT_CONFIDENCE_THRESHOLD,
        "matches_graph_dot_file": Subsetter.DEFAULT_MATCHES_GRAPH_DOT_FILE,
        "estimator": CameraEstimator.DEFAULT_CAMERA_ESTIMATOR,
        "adjuster": CameraAdjuster.DEFAULT_CAMERA_ADJUSTER,
        "refinement_mask": CameraAdjuster.DEFAULT_REFINEMENT_MASK,
        "wave_correct_kind": WaveCorrector.DEFAULT_WAVE_CORRECTION,
        "warper_type": Warper.DEFAULT_WARP_TYPE,
        "low_megapix": Images.Resolution.LOW.value,
        "crop": Cropper.DEFAULT_CROP,
        "compensator": ExposureErrorCompensator.DEFAULT_COMPENSATOR,
        "nr_feeds": ExposureErrorCompensator.DEFAULT_NR_FEEDS,
        "block_size": ExposureErrorCompensator.DEFAULT_BLOCK_SIZE,
        "finder": SeamFinder.DEFAULT_SEAM_FINDER,
        "final_megapix": Images.Resolution.FINAL.value,
        "blender_type": Blender.DEFAULT_BLENDER,
        "blend_strength": Blender.DEFAULT_BLEND_STRENGTH,
        "timelapse": Timelapser.DEFAULT_TIMELAPSE,
        "timelapse_prefix": Timelapser.DEFAULT_TIMELAPSE_PREFIX,
    }

    def __init__(self, **kwargs):
        self.initialize_stitcher(**kwargs)

    def initialize_stitcher(self, **kwargs):
        self.settings = self.DEFAULT_SETTINGS.copy()
        self.validate_kwargs(kwargs)
        self.kwargs = kwargs
        self.settings.update(kwargs)
        self._alignment_correction = (0.0, 0.0)

        args = SimpleNamespace(**self.settings)
        self.medium_megapix = args.medium_megapix
        self.low_megapix = args.low_megapix
        self.final_megapix = args.final_megapix
        if args.detector in ("orb", "sift"):
            self.detector = FeatureDetector(args.detector, nfeatures=args.nfeatures)
        else:
            self.detector = FeatureDetector(args.detector)
        match_conf = FeatureMatcher.get_match_conf(args.match_conf, args.detector)
        self.matcher = FeatureMatcher(
            args.matcher_type,
            args.range_width,
            try_use_gpu=args.try_use_gpu,
            match_conf=match_conf,
        )
        self.subsetter = Subsetter(
            args.confidence_threshold, args.matches_graph_dot_file
        )
        self.megapixels = args.megapixels

        self.megapixel_options = {
            '4': Path(os.path.expanduser('~/stitching/calibration/4mp/config.json')),
            '16': Path(os.path.expanduser('~/stitching/calibration/16mp/config.json')),
            '64': Path(os.path.expanduser('~/stitching/calibration/64mp/config.json'))
        }

        self.camera_estimator = CameraEstimator(args.estimator)
        self.camera_adjuster = CameraAdjuster(
            args.adjuster, args.refinement_mask, args.confidence_threshold
        )
        self.wave_corrector = WaveCorrector(args.wave_correct_kind)
        self.warper = Warper(args.warper_type)
        self.cropper = Cropper(args.crop)
        self.compensator = ExposureErrorCompensator(
            args.compensator, args.nr_feeds, args.block_size
        )
        self.seam_finder = SeamFinder(args.finder)
        self.blender = Blender(args.blender_type, args.blend_strength)
        self.timelapser = Timelapser(args.timelapse, args.timelapse_prefix)

        if args.calibrate is True:
            self.run_calibration = True
            self.cameras = None
            self.cameras_registered = False
            self.calibration_file = args.calibration_file
        else:
            # Check if calibration file exists, and if it
            if args.calibration_file is not None and args.calibration_file != "":
                if not os.path.isabs(args.calibration_file):
                    fp = os.path.expanduser(
                        f"~/stitching/calibration/{self.megapixels}mp/{args.calibration_file}"
                    )
                else:
                    fp = args.calibration_file
            else:
                fp = self.megapixel_options[self.megapixels]
            file_exists = os.path.exists(fp)
            self.cameras = []
            if file_exists:
                with open(fp, 'r') as f:
                    data = json.load(f)

                    left_params = data['left']
                    right_params = data['right']

                    left_cam = CameraParams()
                    right_cam = CameraParams()
                    self.cameras = [self.setup_cam(left_cam, left_params), self.setup_cam(right_cam, right_params)]

                self.cameras_registered = True
                self.run_calibration = False
                self.estimate_scale(self.cameras)

            else:
                self.cameras = None
                self.cameras_registered = False
                self.run_calibration = True
                self.calibration_file = args.calibration_file

    def setup_cam(self, cam, cam_config):
        cam.aspect = cam_config['aspect']
        cam.focal = cam_config['focal']
        cam.ppx = cam_config['ppx']
        cam.ppy = cam_config['ppy']
        cam.t = np.array(cam_config['t'], dtype=np.float32)
        cam.R = np.array(cam_config['R'], dtype=np.float32)

        return cam

    def stitch_verbose(self, images, feature_masks=[], verbose_dir=None):
        return verbose_stitching(self, images, feature_masks, verbose_dir)

    def calibrate(self, feature_masks):

        imgs = self.resize_medium_resolution()
        features = self.find_features(imgs, feature_masks)
        matches = self.match_features(features)
        imgs, features, matches = self.subset(imgs, features, matches)
        cameras = self.estimate_camera_parameters(features, matches)
        cameras = self.refine_camera_parameters(features, matches, cameras)
        cameras = self.perform_wave_correction(cameras)
        self.estimate_scale(cameras)
        self.cameras = cameras
        
        camera_dict = {}

        for idx, camera in enumerate(cameras):

            if idx == 0:
                cam = 'left'
            else:
                cam = 'right'

            camera_dict[cam] = {
                'aspect': camera.aspect, 
                'focal': camera.focal, 
                'ppx': camera.ppx, 
                'ppy': camera.ppy,
                't': camera.t,
                'R': camera.R
            }

        # save to the right calibration file
        if self.calibration_file is not None and self.calibration_file != "":
            if not os.path.isabs(self.calibration_file):
                fp = os.path.expanduser(
                    f"~/stitching/calibration/{self.megapixels}mp/{self.calibration_file}"
                )
            else:
                fp = self.calibration_file
        else:
            fp = self.megapixel_options[self.megapixels]

        with open(fp, 'w') as f:
            json.dump(camera_dict, f, default=convert)

        self.cameras_registered = True
        self.run_calibration = False

    def stitch(self, images, feature_masks=[]):
        self.images = Images.of(
            images, self.medium_megapix, self.low_megapix, self.final_megapix
        )

        if not self.cameras_registered or self.run_calibration:
            self.calibrate(feature_masks)

        imgs = self.resize_low_resolution()
        imgs, masks, corners, sizes = self.warp_low_resolution(imgs, self.cameras)

        # Refine alignment in the overlap region to correct seam misalignment
        corners = self.refine_overlap_alignment(imgs, masks, corners)

        self.prepare_cropper(imgs, masks, corners, sizes)
        imgs, masks, corners, sizes = self.crop_low_resolution(
            imgs, masks, corners, sizes
        )
        self.estimate_exposure_errors(corners, imgs, masks)
        seam_masks = self.find_seam_masks(imgs, corners, masks)

        imgs = self.resize_final_resolution()
        imgs, masks, corners, sizes = self.warp_final_resolution(imgs, self.cameras)

        # Apply the same alignment correction scaled to final resolution
        corners = self.apply_alignment_correction(corners)

        imgs, masks, corners, sizes = self.crop_final_resolution(
            imgs, masks, corners, sizes
        )
        self.set_masks(masks)
        imgs = self.compensate_exposure_errors(corners, imgs)
        seam_masks = self.resize_seam_masks(seam_masks)

        self.initialize_composition(corners, sizes)
        self.blend_images(imgs, seam_masks, corners)
        return self.create_final_panorama()

    def resize_medium_resolution(self):
        return list(self.images.resize(Images.Resolution.MEDIUM))

    def find_features(self, imgs, feature_masks=[]):
        if len(feature_masks) == 0:
            return self.detector.detect(imgs)
        else:
            feature_masks = Images.of(
                feature_masks, self.medium_megapix, self.low_megapix, self.final_megapix
            )
            feature_masks = list(feature_masks.resize(Images.Resolution.MEDIUM))
            feature_masks = [Images.to_binary(mask) for mask in feature_masks]
            return self.detector.detect_with_masks(imgs, feature_masks)

    def match_features(self, features):
        return self.matcher.match_features(features)

    def subset(self, imgs, features, matches):
        indices = self.subsetter.subset(self.images.names, features, matches)
        imgs = Subsetter.subset_list(imgs, indices)
        features = Subsetter.subset_list(features, indices)
        matches = Subsetter.subset_matches(matches, indices)
        self.images.subset(indices)
        return imgs, features, matches

    def estimate_camera_parameters(self, features, matches):
        return self.camera_estimator.estimate(features, matches)

    def refine_camera_parameters(self, features, matches, cameras):
        return self.camera_adjuster.adjust(features, matches, cameras)

    def perform_wave_correction(self, cameras):
        return self.wave_corrector.correct(cameras)

    def estimate_scale(self, cameras):
        self.warper.set_scale(cameras)
        self._alignment_correction = (0.0, 0.0)

    def refine_overlap_alignment(self, imgs, masks, corners):
        """
        Refine alignment between warped images using feature matching
        in the overlap region. Corrects vertical/horizontal misalignment
        caused by imprecise calibration parameters.

        Computes the correction at low resolution and stores it for
        scaling to final resolution.
        """
        self._alignment_correction = (0.0, 0.0)

        if len(imgs) != 2:
            return corners

        left_img, right_img = imgs[0], imgs[1]
        left_corner, right_corner = corners[0], corners[1]

        # Find overlap region in panorama coordinates
        left_x_end = left_corner[0] + left_img.shape[1]
        right_x_end = right_corner[0] + right_img.shape[1]
        overlap_x_start = max(left_corner[0], right_corner[0])
        overlap_x_end = min(left_x_end, right_x_end)

        if overlap_x_end <= overlap_x_start + 5:
            return corners

        # Common Y range
        left_y_end = left_corner[1] + left_img.shape[0]
        right_y_end = right_corner[1] + right_img.shape[0]
        common_y_start = max(left_corner[1], right_corner[1])
        common_y_end = min(left_y_end, right_y_end)

        if common_y_end <= common_y_start + 5:
            return corners

        # Extract overlap regions in local image coordinates
        l_x1 = overlap_x_start - left_corner[0]
        l_x2 = overlap_x_end - left_corner[0]
        l_y1 = common_y_start - left_corner[1]
        l_y2 = common_y_end - left_corner[1]

        r_x1 = overlap_x_start - right_corner[0]
        r_x2 = overlap_x_end - right_corner[0]
        r_y1 = common_y_start - right_corner[1]
        r_y2 = common_y_end - right_corner[1]

        left_overlap = left_img[l_y1:l_y2, l_x1:l_x2]
        right_overlap = right_img[r_y1:r_y2, r_x1:r_x2]

        # Ensure same size
        h = min(left_overlap.shape[0], right_overlap.shape[0])
        w = min(left_overlap.shape[1], right_overlap.shape[1])
        left_overlap = left_overlap[:h, :w]
        right_overlap = right_overlap[:h, :w]

        if h < 10 or w < 10:
            return corners

        # Try feature-based alignment first, fall back to phase correlation
        dx, dy = self._feature_based_alignment(left_overlap, right_overlap)

        if dx is None or dy is None:
            dx, dy = self._phase_correlation_alignment(left_overlap, right_overlap)

        if dx is None or dy is None:
            return corners

        self._alignment_correction = (dx, dy)

        # Apply correction to right image corner
        new_corners = list(corners)
        new_corners[1] = (corners[1][0] + round(dx), corners[1][1] + round(dy))
        return new_corners

    def _feature_based_alignment(self, left_overlap, right_overlap):
        """
        Use SIFT feature matching to find the precise offset between
        two overlap regions. Returns (dx, dy) or (None, None) on failure.
        """
        # Convert to grayscale
        if len(left_overlap.shape) == 3:
            left_gray = cv.cvtColor(left_overlap, cv.COLOR_BGR2GRAY)
            right_gray = cv.cvtColor(right_overlap, cv.COLOR_BGR2GRAY)
        else:
            left_gray = left_overlap
            right_gray = right_overlap

        # Detect features - try SIFT first, fall back to ORB
        try:
            detector = cv.SIFT_create(nfeatures=1000)
            use_flann = True
        except cv.error:
            detector = cv.ORB_create(nfeatures=1000)
            use_flann = False

        kp1, des1 = detector.detectAndCompute(left_gray, None)
        kp2, des2 = detector.detectAndCompute(right_gray, None)

        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            return None, None

        # Match features
        if use_flann:
            index_params = dict(algorithm=1, trees=5)  # FLANN_INDEX_KDTREE
            search_params = dict(checks=50)
            matcher = cv.FlannBasedMatcher(index_params, search_params)
        else:
            matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
        matches = matcher.knnMatch(des1, des2, k=2)

        # Lowe's ratio test
        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)

        if len(good_matches) < 4:
            return None, None

        # Compute translation from matched points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])

        # Use RANSAC to find inlier translation
        offsets = dst_pts - src_pts
        # Find translation using RANSAC on the offsets
        best_dx, best_dy, best_inliers = 0, 0, 0
        threshold = 2.0  # pixels

        for i in range(min(len(offsets), 200)):
            candidate_dx = offsets[i, 0]
            candidate_dy = offsets[i, 1]
            errors = np.sqrt(
                (offsets[:, 0] - candidate_dx) ** 2
                + (offsets[:, 1] - candidate_dy) ** 2
            )
            inliers = np.sum(errors < threshold)
            if inliers > best_inliers:
                best_inliers = inliers
                best_dx = candidate_dx
                best_dy = candidate_dy

        if best_inliers < 4:
            return None, None

        # Refine using median of inliers
        errors = np.sqrt(
            (offsets[:, 0] - best_dx) ** 2 + (offsets[:, 1] - best_dy) ** 2
        )
        inlier_mask = errors < threshold
        dx = -np.median(offsets[inlier_mask, 0])
        dy = -np.median(offsets[inlier_mask, 1])

        return dx, dy

    def _phase_correlation_alignment(self, left_overlap, right_overlap):
        """
        Use phase correlation to find the translation offset between
        two overlap regions. Returns (dx, dy) or (None, None) on failure.
        """
        if len(left_overlap.shape) == 3:
            left_gray = cv.cvtColor(left_overlap, cv.COLOR_BGR2GRAY)
            right_gray = cv.cvtColor(right_overlap, cv.COLOR_BGR2GRAY)
        else:
            left_gray = left_overlap
            right_gray = right_overlap

        left_float = left_gray.astype(np.float64)
        right_float = right_gray.astype(np.float64)

        h, w = left_float.shape[:2]
        hann = cv.createHanningWindow((w, h), cv.CV_64F)

        (dx, dy), response = cv.phaseCorrelate(
            left_float * hann, right_float * hann
        )

        if response > 0.05:
            return dx, dy

        return None, None

    def apply_alignment_correction(self, corners):
        """
        Apply the stored alignment correction scaled from low to final resolution.
        """
        dx_low, dy_low = self._alignment_correction
        if abs(dx_low) < 0.01 and abs(dy_low) < 0.01:
            return corners

        scale = self.images.get_ratio(
            Images.Resolution.LOW, Images.Resolution.FINAL
        )

        dx_final = dx_low * scale
        dy_final = dy_low * scale

        new_corners = list(corners)
        if len(new_corners) >= 2:
            new_corners[1] = (
                corners[1][0] + round(dx_final),
                corners[1][1] + round(dy_final),
            )
        return new_corners

    def resize_low_resolution(self, imgs=None):
        return list(self.images.resize(Images.Resolution.LOW, imgs))

    def warp_low_resolution(self, imgs, cameras):
        sizes = self.images.get_scaled_img_sizes(Images.Resolution.LOW)
        camera_aspect = self.images.get_ratio(
            Images.Resolution.MEDIUM, Images.Resolution.LOW
        )
        imgs, masks, corners, sizes = self.warp(imgs, cameras, sizes, camera_aspect)
        return list(imgs), list(masks), corners, sizes

    def warp_final_resolution(self, imgs, cameras):
        sizes = self.images.get_scaled_img_sizes(Images.Resolution.FINAL)
        camera_aspect = self.images.get_ratio(
            Images.Resolution.MEDIUM, Images.Resolution.FINAL
        )
        return self.warp(imgs, cameras, sizes, camera_aspect)

    def warp(self, imgs, cameras, sizes, aspect=1):
        imgs = self.warper.warp_images(imgs, cameras, aspect)
        masks = self.warper.create_and_warp_masks(sizes, cameras, aspect)
        corners, sizes = self.warper.warp_rois(sizes, cameras, aspect)
        return imgs, masks, corners, sizes

    def prepare_cropper(self, imgs, masks, corners, sizes):
        self.cropper.prepare(imgs, masks, corners, sizes)

    def crop_low_resolution(self, imgs, masks, corners, sizes):
        imgs, masks, corners, sizes = self.crop(imgs, masks, corners, sizes)
        return list(imgs), list(masks), corners, sizes

    def crop_final_resolution(self, imgs, masks, corners, sizes):
        lir_aspect = self.images.get_ratio(
            Images.Resolution.LOW, Images.Resolution.FINAL
        )
        return self.crop(imgs, masks, corners, sizes, lir_aspect)

    def crop(self, imgs, masks, corners, sizes, aspect=1):
        masks = self.cropper.crop_images(masks, aspect)
        imgs = self.cropper.crop_images(imgs, aspect)
        corners, sizes = self.cropper.crop_rois(corners, sizes, aspect)
        return imgs, masks, corners, sizes

    def estimate_exposure_errors(self, corners, imgs, masks):
        self.compensator.feed(corners, imgs, masks)

    def find_seam_masks(self, imgs, corners, masks):
        return self.seam_finder.find(imgs, corners, masks)

    def resize_final_resolution(self):
        return self.images.resize(Images.Resolution.FINAL)

    def compensate_exposure_errors(self, corners, imgs):
        for idx, (corner, img) in enumerate(zip(corners, imgs)):
            yield self.compensator.apply(idx, corner, img, self.get_mask(idx))

    def resize_seam_masks(self, seam_masks):
        for idx, seam_mask in enumerate(seam_masks):
            yield SeamFinder.resize(seam_mask, self.get_mask(idx))

    def set_masks(self, mask_generator):
        self.masks = mask_generator
        self.mask_index = -1

    def get_mask(self, idx):
        if idx == self.mask_index + 1:
            self.mask_index += 1
            self.mask = next(self.masks)
            return self.mask
        elif idx == self.mask_index:
            return self.mask
        else:
            raise StitchingError("Invalid Mask Index!")

    def initialize_composition(self, corners, sizes):
        if self.timelapser.do_timelapse:
            self.timelapser.initialize(corners, sizes)
        else:
            self.blender.prepare(corners, sizes)

    def blend_images(self, imgs, masks, corners):
        for idx, (img, mask, corner) in enumerate(zip(imgs, masks, corners)):
            if self.timelapser.do_timelapse:
                self.timelapser.process_and_save_frame(
                    self.images.names[idx], img, corner
                )
            else:
                self.blender.feed(img, mask, corner)

    def create_final_panorama(self):
        if not self.timelapser.do_timelapse:
            panorama, _ = self.blender.blend()
            return panorama

    def validate_kwargs(self, kwargs):
        for arg in kwargs:
            if arg not in self.DEFAULT_SETTINGS:
                raise StitchingError("Invalid Argument: " + arg)


class AffineStitcher(Stitcher):
    AFFINE_DEFAULTS = {
        "estimator": "affine",
        "wave_correct_kind": "no",
        "matcher_type": "affine",
        "adjuster": "affine",
        "warper_type": "affine",
        "compensator": "no",
    }

    DEFAULT_SETTINGS = Stitcher.DEFAULT_SETTINGS.copy()
    DEFAULT_SETTINGS.update(AFFINE_DEFAULTS)

    def initialize_stitcher(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.AFFINE_DEFAULTS and value != self.AFFINE_DEFAULTS[key]:
                warnings.warn(
                    f"You are overwriting an affine default ({key}={self.AFFINE_DEFAULTS[key]}) with another value ({value}). Make sure this is intended",  # noqa: E501
                    StitchingWarning,
                )
        super().initialize_stitcher(**kwargs)
