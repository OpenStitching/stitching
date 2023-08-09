class StitchingError(Exception):
    pass


class CameraAdjustmentError(StitchingError):
    pass


class InvalidMaskIndexError(StitchingError):
    pass


class InvalidArgumentError(StitchingError):
    pass


class NoMatchExceedsThresholdError(StitchingError):
    pass


class HomographyEstimationError(StitchingError):
    pass


class InvalidContourError(StitchingError):
    pass


class RectanglesNotOverlapError(StitchingError):
    pass


class ResolutionMismatchError(StitchingError):
    pass


class InsufficientImagesError(StitchingError):
    pass


class ImageReadError(StitchingError):
    pass
