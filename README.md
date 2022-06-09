# stitching

A Python package for fast and robust Image Stitching. A modularized
and continuing work based on opencv's [stitching
module](https://github.com/opencv/opencv/tree/4.x/modules/stitching)
and the
[stitching_detailed.py](https://github.com/opencv/opencv/blob/4.x/samples/python/stitching_detailed.py)
python command line tool.

![inputs](https://github.com/lukasalexanderweber/stitching_tutorial/blob/master/docs/static_files/inputs.png?raw=true)

![result](https://github.com/lukasalexanderweber/stitching_tutorial/blob/master/docs/static_files/panorama.png?raw=true)

## Installation

Use pip to install stitching from
[PyPI](https://pypi.org/project/stitching/).

```bash
pip install stitching
```

## Usage

```python
import stitching

stitcher = stitching.Stitcher()
panorama = stitcher.stitch(["img1.jpg", "img2.jpg", "img3.jpg"])

```

or using the command line interface ([cli](https://github.com/lukasalexanderweber/stitching/tree/main/stitching/cli/stitch.py)) which is available after installation

```bash
stitch -h
stitch img1.jpg img2.jpg img3.jpg
```

## Tutorial

This package provides utility functions to deeply analyse what's
happening behind the stitching. A tutorial was created as [Jupyter
Notebook](https://github.com/lukasalexanderweber/stitching_tutorial). The
preview is
[here](https://github.com/lukasalexanderweber/stitching_tutorial/blob/master/docs/Stitching%20Tutorial.md).

You can e.g. visualize the RANSAC matches between the images or the
seam lines where the images are blended:

![matches1](https://github.com/lukasalexanderweber/stitching_tutorial/blob/master/docs/static_files/matches1.png?raw=true)
![matches2](https://github.com/lukasalexanderweber/stitching_tutorial/blob/master/docs/static_files/matches2.png?raw=true)
![seams1](https://github.com/lukasalexanderweber/stitching_tutorial/blob/master/docs/static_files/seams1.png?raw=true)
![seams2](https://github.com/lukasalexanderweber/stitching_tutorial/blob/master/docs/static_files/seams2.png?raw=true)

## Literature

This package was developed and used for our paper [Automatic stitching
of fragmented construction plans of hydraulic
structures](https://doi.org/10.1002/bate.202200010 "Automatic
stitching of fragmented construction plans of hydraulic structures")

## Contributing

Pull requests are welcome. For major changes, please open an issue
first to discuss what you would like to change.

Please make sure to update tests as appropriate.

Run tests using

```bash
python -m unittest
```

Build with

```bash
python -m build
```

Please make sure to lint all pull requests.

Lint the changed files

```bash
pre-commit install && pre-commit run
```

## License

[Apache License
2.0](https://github.com/lukasalexanderweber/lir/blob/main/LICENSE)
