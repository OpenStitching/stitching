import os

import cv2 as cv

from .images import Images
from .seam_finder import SeamFinder
from .timelapser import Timelapser


def verbose_stitching(stitcher, images, feature_masks=[], verbose_dir=None):
    _dir = "." if verbose_dir is None else verbose_dir

    with open(verbose_output(_dir, "00_stitcher.txt"), "w") as file:
        file.write(type(stitcher).__name__ + "(**" + str(stitcher.kwargs) + ")")

    images = Images.of(
        images, stitcher.medium_megapix, stitcher.low_megapix, stitcher.final_megapix
    )

    # Resize Images
    imgs = list(images.resize(Images.Resolution.MEDIUM))

    # Find Features
    finder = stitcher.detector
    features = stitcher.find_features(imgs, feature_masks)
    for idx, img_features in enumerate(features):
        img_with_features = finder.draw_keypoints(imgs[idx], img_features)
        write_verbose_result(_dir, f"01_features_img{idx + 1}.jpg", img_with_features)

    # Match Features
    matcher = stitcher.matcher
    matches = matcher.match_features(features)

    # Subset
    subsetter = stitcher.subsetter

    all_relevant_matches = list(
        matcher.draw_matches_matrix(
            imgs,
            features,
            matches,
            conf_thresh=subsetter.confidence_threshold,
            inliers=True,
            matchColor=(0, 255, 0),
        )
    )
    for idx1, idx2, img in all_relevant_matches:
        write_verbose_result(
            _dir, f"02_matches_img{idx1 + 1}_to_img{idx2 + 1}.jpg", img
        )

    # Subset
    subsetter = stitcher.subsetter
    subsetter.save_file = verbose_output(_dir, "03_matches_graph.txt")
    subsetter.save_matches_graph_dot_file(images.names, matches)

    indices = subsetter.get_indices_to_keep(features, matches)

    imgs = subsetter.subset_list(imgs, indices)
    features = subsetter.subset_list(features, indices)
    matches = subsetter.subset_matches(matches, indices)
    images.subset(indices)

    # Camera Estimation, Adjustion and Correction
    camera_estimator = stitcher.camera_estimator
    camera_adjuster = stitcher.camera_adjuster
    wave_corrector = stitcher.wave_corrector

    cameras = camera_estimator.estimate(features, matches)
    cameras = camera_adjuster.adjust(features, matches, cameras)
    cameras = wave_corrector.correct(cameras)

    # Warp Images
    low_imgs = list(images.resize(Images.Resolution.LOW, imgs))
    imgs = None  # free memory

    warper = stitcher.warper
    warper.set_scale(cameras)

    low_sizes = images.get_scaled_img_sizes(Images.Resolution.LOW)
    camera_aspect = images.get_ratio(Images.Resolution.MEDIUM, Images.Resolution.LOW)

    low_imgs = list(warper.warp_images(low_imgs, cameras, camera_aspect))
    low_masks = list(warper.create_and_warp_masks(low_sizes, cameras, camera_aspect))
    low_corners, low_sizes = warper.warp_rois(low_sizes, cameras, camera_aspect)

    final_sizes = images.get_scaled_img_sizes(Images.Resolution.FINAL)
    camera_aspect = images.get_ratio(Images.Resolution.MEDIUM, Images.Resolution.FINAL)

    final_imgs = list(images.resize(Images.Resolution.FINAL))
    final_imgs = list(warper.warp_images(final_imgs, cameras, camera_aspect))
    final_masks = list(
        warper.create_and_warp_masks(final_sizes, cameras, camera_aspect)
    )
    final_corners, final_sizes = warper.warp_rois(final_sizes, cameras, camera_aspect)

    for idx, warped_img in enumerate(final_imgs):
        write_verbose_result(_dir, f"04_warped_img{idx + 1}.jpg", warped_img)

    # Excursion: Timelapser
    timelapser = Timelapser("as_is")
    timelapser.initialize(final_corners, final_sizes)

    for idx, (img, corner) in enumerate(zip(final_imgs, final_corners)):
        timelapser.process_frame(img, corner)
        frame = timelapser.get_frame()
        write_verbose_result(_dir, f"05_timelapse_img{idx + 1}.jpg", frame)

    # Crop
    cropper = stitcher.cropper

    if cropper.do_crop:
        mask = cropper.estimate_panorama_mask(
            low_imgs, low_masks, low_corners, low_sizes
        )
        write_verbose_result(_dir, "06_estimated_mask_to_crop.jpg", mask)

        lir = cropper.estimate_largest_interior_rectangle(mask)

        lir_to_crop = lir.draw_on(mask, size=2)
        write_verbose_result(_dir, "06_lir.jpg", lir_to_crop)

        low_corners = cropper.get_zero_center_corners(low_corners)
        cropper.prepare(low_imgs, low_masks, low_corners, low_sizes)

        low_masks = list(cropper.crop_images(low_masks))
        low_imgs = list(cropper.crop_images(low_imgs))
        low_corners, low_sizes = cropper.crop_rois(low_corners, low_sizes)

        lir_aspect = images.get_ratio(Images.Resolution.LOW, Images.Resolution.FINAL)
        final_masks = list(cropper.crop_images(final_masks, lir_aspect))
        final_imgs = list(cropper.crop_images(final_imgs, lir_aspect))
        final_corners, final_sizes = cropper.crop_rois(
            final_corners, final_sizes, lir_aspect
        )

        timelapser = Timelapser("as_is")
        timelapser.initialize(final_corners, final_sizes)

        for idx, (img, corner) in enumerate(zip(final_imgs, final_corners)):
            timelapser.process_frame(img, corner)
            frame = timelapser.get_frame()
            write_verbose_result(_dir, f"07_timelapse_cropped_img{idx + 1}.jpg", frame)

    # Seam Masks
    seam_finder = stitcher.seam_finder

    seam_masks = seam_finder.find(low_imgs, low_corners, low_masks)
    seam_masks = [
        seam_finder.resize(seam_mask, mask)
        for seam_mask, mask in zip(seam_masks, final_masks)
    ]
    seam_masks_plots = [
        SeamFinder.draw_seam_mask(img, seam_mask)
        for img, seam_mask in zip(final_imgs, seam_masks)
    ]

    for idx, seam_mask in enumerate(seam_masks_plots):
        write_verbose_result(_dir, f"08_seam_mask{idx + 1}.jpg", seam_mask)

    # Exposure Error Compensation
    compensator = stitcher.compensator

    compensator.feed(low_corners, low_imgs, low_masks)

    compensated_imgs = [
        compensator.apply(idx, corner, img, mask)
        for idx, (img, mask, corner) in enumerate(
            zip(final_imgs, final_masks, final_corners)
        )
    ]

    for idx, compensated_img in enumerate(compensated_imgs):
        write_verbose_result(_dir, f"08_compensated{idx + 1}.jpg", compensated_img)

    # Blending
    blender = stitcher.blender
    blender.prepare(final_corners, final_sizes)
    for img, mask, corner in zip(compensated_imgs, seam_masks, final_corners):
        blender.feed(img, mask, corner)
    panorama, _ = blender.blend()

    write_verbose_result(_dir, "09_result.jpg", panorama)

    blended_seam_masks = seam_finder.blend_seam_masks(
        seam_masks, final_corners, final_sizes
    )
    with_seam_lines = seam_finder.draw_seam_lines(
        panorama, blended_seam_masks, linesize=3
    )
    with_seam_polygons = seam_finder.draw_seam_polygons(panorama, blended_seam_masks)

    write_verbose_result(_dir, "09_result_with_seam_lines.jpg", with_seam_lines)
    write_verbose_result(_dir, "09_result_with_seam_polygons.jpg", with_seam_polygons)

    return panorama


def write_verbose_result(dir_name, img_name, img):
    cv.imwrite(verbose_output(dir_name, img_name), img)


def verbose_output(dir_name, file):
    return os.path.join(dir_name, file)
