FROM python:3.11

RUN mkdir /stitching
COPY . /stitching

# build and install stitching package
WORKDIR /stitching
RUN pip install build
# we use opencv headless within docker, otherwise we get errors
RUN sed -i 's/opencv-python/opencv-python-headless/g' setup.cfg
RUN python -m build
RUN pip install ./dist/stitching-*.whl

# compile largestinteriorrectangle (JIT)
RUN python -c "import largestinteriorrectangle"

# provide the entrypoint, users need to mount a volume to /data
WORKDIR /data
ENTRYPOINT ["stitch"]
CMD ["-h"]
