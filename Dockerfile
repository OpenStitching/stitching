FROM python:3.11 AS builder

WORKDIR /stitching
COPY . .
RUN pip install build
# we use opencv headless within docker, otherwise we get errors
RUN sed -i 's/opencv-python/opencv-python-headless/g' setup.cfg
RUN python -m build

FROM python:3.11-slim

WORKDIR /stitching
COPY --from=builder /stitching/dist/stitching-*.whl .
RUN pip install stitching-*.whl

# compile largestinteriorrectangle (JIT)
RUN python -c "import largestinteriorrectangle"

# provide the entrypoint, users need to mount a volume to /data
WORKDIR /data
ENTRYPOINT ["stitch"]
CMD ["-h"]
