import unittest
import os
import time
import tracemalloc

from .context import Stitcher
from .stitching_detailed import main


class TestStitcher(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        TEST_DIR = os.path.abspath(os.path.dirname(__file__))
        os.chdir(os.path.join(TEST_DIR, "testdata"))

    @unittest.skip("skip performance test (not needed in every run)")
    def test_performance(self):

        print("Run new Stitcher class:")

        start = time.time()
        tracemalloc.start()

        stitcher = Stitcher(final_megapix=3, crop=False)
        stitcher.stitch(["boat5.jpg", "boat2.jpg",
                         "boat3.jpg", "boat4.jpg",
                         "boat1.jpg", "boat6.jpg"])

        _, peak_memory = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        end = time.time()
        time_needed = end - start

        print(f"Peak was {peak_memory / 10**6} MB")
        print(f"Time was {time_needed} s")

        print("Run original stitching_detailed.py:")

        start = time.time()
        tracemalloc.start()

        main()

        _, peak_memory_detailed = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        end = time.time()
        time_needed_detailed = end - start

        print(f"Peak was {peak_memory_detailed / 10**6} MB")
        print(f"Time was {time_needed_detailed} s")

        self.assertLessEqual(peak_memory / 10**6,
                             peak_memory_detailed / 10**6)
        uncertainty_based_on_run = 0.25
        self.assertLessEqual(time_needed - uncertainty_based_on_run,
                             time_needed_detailed)


def starttest():
    unittest.main()


if __name__ == "__main__":
    starttest()
