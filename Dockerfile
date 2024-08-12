FROM python:3.11 AS builder

WORKDIR /stitching

COPY setup.cfg pyproject.toml README.md .
COPY stitching/ ./stitching/

RUN pip install --no-cache-dir build && \
    sed -i 's/opencv-python/opencv-python-headless/g' setup.cfg && \
    python -m build

FROM python:3.11-slim

WORKDIR /stitching
COPY --from=builder /stitching/dist/stitching-*.whl .
RUN pip install --no-cache-dir stitching-*.whl && \
    rm stitching-*.whl

# compile largestinteriorrectangle (JIT)
RUN python -c "import largestinteriorrectangle"

# provide the entrypoint, users need to mount a volume to /data
WORKDIR /data
ENTRYPOINT ["stitch"]
CMD ["-h"]
