import unittest

import numpy as np

from context import (
    Blender,
    CameraAdjuster,
    CameraEstimator,
    Cropper,
    ExposureErrorCompensator,
    FeatureDetector,
    FeatureMatcher,
    Images,
    SeamFinder,
    Subsetter,
    Timelapser,
    Warper,
    WaveCorrector,
    test_input,
    tutorial_output,
    write_tutorial_result,
)


class TestStitcherTutorial(unittest.TestCase):
    def test_tutorial(self):
        images = Images.of([test_input("weir*")])

        # Resize Images
        medium_imgs = list(images.resize(Images.Resolution.MEDIUM))
        low_imgs = list(images.resize(Images.Resolution.LOW))
        final_imgs = list(images.resize(Images.Resolution.FINAL))

        original_size = images.sizes[0]  # noqa F401
        medium_size = Images.get_image_size(medium_imgs[0])  # noqa F401
        low_size = Images.get_image_size(low_imgs[0])  # noqa F401
        final_size = Images.get_image_size(final_imgs[0])  # noqa F401

        # Find Features
        finder = FeatureDetector()
        features = [finder.detect_features(img) for img in medium_imgs]
        for idx, img_features in enumerate(features):
            img_with_features = finder.draw_keypoints(medium_imgs[idx], img_features)
            write_tutorial_result(
                f"tutorial_01_features_img{idx+1}.jpg", img_with_features
            )

        # Match Features
        matcher = FeatureMatcher()
        matches = matcher.match_features(features)

        matcher.get_confidence_matrix(matches)

        all_relevant_matches = list(
            matcher.draw_matches_matrix(
                medium_imgs,
                features,
                matches,
                conf_thresh=1,
                inliers=True,
                matchColor=(0, 255, 0),
            )
        )
        for idx1, idx2, img in all_relevant_matches:
            write_tutorial_result(
                f"tutorial_02_matches_img{idx1+1}_to_img{idx2+1}.jpg", img
            )

        # Subset
        graph_file = tutorial_output("tutorial_03_matches_graph.txt")
        subsetter = Subsetter(matches_graph_dot_file=graph_file)
        subsetter.save_matches_graph_dot_file(images.names, matches)

        indices = subsetter.get_indices_to_keep(features, matches)

        medium_imgs = subsetter.subset_list(medium_imgs, indices)
        low_imgs = subsetter.subset_list(low_imgs, indices)
        final_imgs = subsetter.subset_list(final_imgs, indices)
        features = subsetter.subset_list(features, indices)
        matches = subsetter.subset_matches(matches, indices)
        images.subset(indices)

        # Camera Estimation, Adjustion and Correction
        camera_estimator = CameraEstimator()
        camera_adjuster = CameraAdjuster()
        wave_corrector = WaveCorrector()

        cameras = camera_estimator.estimate(features, matches)
        cameras = camera_adjuster.adjust(features, matches, cameras)
        cameras = wave_corrector.correct(cameras)

        # Warp Images
        warper = Warper()
        warper.set_scale(cameras)

        low_sizes = images.get_scaled_img_sizes(Images.Resolution.LOW)
        camera_aspect = images.get_ratio(
            Images.Resolution.MEDIUM, Images.Resolution.LOW
        )

        warped_low_imgs = list(warper.warp_images(low_imgs, cameras, camera_aspect))
        warped_low_masks = list(
            warper.create_and_warp_masks(low_sizes, cameras, camera_aspect)
        )
        low_corners, low_sizes = warper.warp_rois(low_sizes, cameras, camera_aspect)

        final_sizes = images.get_scaled_img_sizes(Images.Resolution.FINAL)
        camera_aspect = images.get_ratio(
            Images.Resolution.MEDIUM, Images.Resolution.FINAL
        )

        warped_final_imgs = list(warper.warp_images(final_imgs, cameras, camera_aspect))
        warped_final_masks = list(
            warper.create_and_warp_masks(final_sizes, cameras, camera_aspect)
        )
        final_corners, final_sizes = warper.warp_rois(
            final_sizes, cameras, camera_aspect
        )

        for idx, warped_img in enumerate(warped_final_imgs):
            write_tutorial_result(f"tutorial_04_warped_img{idx+1}.jpg", warped_img)

        # Excursion: Timelapser
        timelapser = Timelapser("as_is")
        timelapser.initialize(final_corners, final_sizes)

        for idx, (img, corner) in enumerate(zip(warped_final_imgs, final_corners)):
            timelapser.process_frame(img, corner)
            frame = timelapser.get_frame()
            write_tutorial_result(f"tutorial_05_timelapse_img{idx+1}.jpg", frame)

        # Crop
        cropper = Cropper()

        mask = cropper.estimate_panorama_mask(
            warped_low_imgs, warped_low_masks, low_corners, low_sizes
        )
        lir = cropper.estimate_largest_interior_rectangle(mask)

        lir_to_crop = lir.draw_on(mask, size=2)
        write_tutorial_result("tutorial_06_crop.jpg", lir_to_crop)

        low_corners = cropper.get_zero_center_corners(low_corners)
        rectangles = cropper.get_rectangles(low_corners, low_sizes)
        overlap = cropper.get_overlap(rectangles[1], lir)
        intersection = cropper.get_intersection(rectangles[1], overlap)  # noqa F401

        cropper.prepare(warped_low_imgs, warped_low_masks, low_corners, low_sizes)

        cropped_low_masks = list(cropper.crop_images(warped_low_masks))
        cropped_low_imgs = list(cropper.crop_images(warped_low_imgs))
        low_corners, low_sizes = cropper.crop_rois(low_corners, low_sizes)

        lir_aspect = images.get_ratio(Images.Resolution.LOW, Images.Resolution.FINAL)
        cropped_final_masks = list(cropper.crop_images(warped_final_masks, lir_aspect))
        cropped_final_imgs = list(cropper.crop_images(warped_final_imgs, lir_aspect))
        final_corners, final_sizes = cropper.crop_rois(
            final_corners, final_sizes, lir_aspect
        )

        timelapser = Timelapser("as_is")
        timelapser.initialize(final_corners, final_sizes)

        for idx, (img, corner) in enumerate(zip(cropped_final_imgs, final_corners)):
            timelapser.process_frame(img, corner)
            frame = timelapser.get_frame()
            write_tutorial_result(
                f"tutorial_07_timelapse_cropped_img{idx+1}.jpg", frame
            )

        # Seam Masks
        seam_finder = SeamFinder()

        seam_masks = seam_finder.find(cropped_low_imgs, low_corners, cropped_low_masks)
        seam_masks = [
            seam_finder.resize(seam_mask, mask)
            for seam_mask, mask in zip(seam_masks, cropped_final_masks)
        ]
        seam_masks_plots = [
            SeamFinder.draw_seam_mask(img, seam_mask)
            for img, seam_mask in zip(cropped_final_imgs, seam_masks)
        ]

        for idx, seam_mask in enumerate(seam_masks_plots):
            write_tutorial_result(f"tutorial_08_seam_mask{idx+1}.jpg", seam_mask)

        # Exposure Error Compensation
        compensator = ExposureErrorCompensator()

        compensator.feed(low_corners, cropped_low_imgs, cropped_low_masks)

        compensated_imgs = [
            compensator.apply(idx, corner, img, mask)
            for idx, (img, mask, corner) in enumerate(
                zip(cropped_final_imgs, cropped_final_masks, final_corners)
            )
        ]

        for idx, compensated_img in enumerate(compensated_imgs):
            write_tutorial_result(
                f"tutorial_08_compensated{idx+1}.jpg", compensated_img
            )

        # Blending
        blender = Blender()
        blender.prepare(final_corners, final_sizes)
        for img, mask, corner in zip(compensated_imgs, seam_masks, final_corners):
            blender.feed(img, mask, corner)
        panorama, _ = blender.blend()

        write_tutorial_result("tutorial_09_result.jpg", panorama)

        blended_seam_masks = seam_finder.blend_seam_masks(
            seam_masks, final_corners, final_sizes
        )
        with_seam_lines = seam_finder.draw_seam_lines(
            panorama, blended_seam_masks, linesize=3
        )
        with_seam_polygons = seam_finder.draw_seam_polygons(
            panorama, blended_seam_masks
        )

        write_tutorial_result(
            "tutorial_09_result_with_seam_lines.jpg", with_seam_lines
        )
        write_tutorial_result(
            "tutorial_09_result_with_seam_polygons.jpg", with_seam_polygons
        )

        # Check only that the result is correct.
        # Mostly this test is for checking that no error occurs during the tutorial.
        max_image_shape_derivation = 25
        np.testing.assert_allclose(
            panorama.shape[:2], (673, 2636), atol=max_image_shape_derivation
        )


def start_test():
    unittest.main()


if __name__ == "__main__":
    start_test()
