import time
import tracemalloc
import unittest

from .context import (
    Blender,
    CameraAdjuster,
    CameraEstimator,
    ExposureErrorCompensator,
    FeatureDetector,
    FeatureMatcher,
    Images,
    SeamFinder,
    Stitcher,
    Subsetter,
    Warper,
    WaveCorrector,
    test_input,
)
from .stitching_detailed import main


class TestStitcher(unittest.TestCase):
    def test_performance(self):
        test_imgs = [
            test_input("boat5.jpg"),
            test_input("boat2.jpg"),
            test_input("boat3.jpg"),
            test_input("boat4.jpg"),
            test_input("boat1.jpg"),
            test_input("boat6.jpg"),
        ]

        # print("Run Stitcher:")

        start = time.time()
        tracemalloc.start()

        stitcher = Stitcher(crop=False)
        stitcher.stitch(test_imgs)

        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        end = time.time()
        time_needed = end - start

        # print(f"Peak was {peak_memory / 10**6} MB")
        # print(f"Time was {time_needed} s")

        # print("Run original stitching_detailed.py:")

        kwargs = {
            "img_names": test_imgs,
            "try_cuda": False,
            "work_megapix": Images.Resolution.MEDIUM.value,
            "features": FeatureDetector.DEFAULT_DETECTOR,
            "matcher": FeatureMatcher.DEFAULT_MATCHER,
            "estimator": CameraEstimator.DEFAULT_CAMERA_ESTIMATOR,
            "match_conf": None,
            "conf_thresh": Subsetter.DEFAULT_CONFIDENCE_THRESHOLD,
            "ba": CameraAdjuster.DEFAULT_CAMERA_ADJUSTER,
            "ba_refine_mask": CameraAdjuster.DEFAULT_REFINEMENT_MASK,
            "wave_correct": WaveCorrector.DEFAULT_WAVE_CORRECTION,
            "save_graph": Subsetter.DEFAULT_MATCHES_GRAPH_DOT_FILE,
            "warp": Warper.DEFAULT_WARP_TYPE,
            "seam_megapix": Images.Resolution.LOW.value,
            "seam": SeamFinder.DEFAULT_SEAM_FINDER,
            "compose_megapix": Images.Resolution.FINAL.value,
            "expos_comp": ExposureErrorCompensator.DEFAULT_COMPENSATOR,
            "expos_comp_nr_feeds": ExposureErrorCompensator.DEFAULT_NR_FEEDS,
            "expos_comp_nr_filtering": 2,  # not used in stitching_detailed
            "expos_comp_block_size": ExposureErrorCompensator.DEFAULT_BLOCK_SIZE,
            "blend": Blender.DEFAULT_BLENDER,
            "blend_strength": Blender.DEFAULT_BLEND_STRENGTH,
            "timelapse": None,  # not backwards compatible "no" != None
            "rangewidth": FeatureMatcher.DEFAULT_RANGE_WIDTH,
        }

        start = time.time()
        tracemalloc.start()

        main(**kwargs)

        _, peak_memory_detailed = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        end = time.time()
        time_needed_detailed = end - start

        # print(f"Peak was {peak_memory_detailed / 10**6} MB")
        # print(f"Time was {time_needed_detailed} s")

        # We use 10% less memory than the original approach
        allowed_deviation_in_percent = 10
        allowed_deviation = peak_memory_detailed / 100 * allowed_deviation_in_percent
        self.assertLessEqual(peak_memory, peak_memory_detailed + allowed_deviation)

        # We allow ourself to be a maximum of 5% slower
        allowed_deviation_in_percent = 5
        allowed_deviation = time_needed / 100 * allowed_deviation_in_percent
        self.assertLessEqual(time_needed - allowed_deviation, time_needed_detailed)


def starttest():
    unittest.main()


if __name__ == "__main__":
    starttest()
