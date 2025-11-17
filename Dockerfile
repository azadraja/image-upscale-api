# --- STAGE 1: Builder ---
FROM golang:1.23-bookworm AS builder

# Build prerequisites
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git wget unzip yasm pkg-config \
    libtbb-dev libjpeg-dev libpng-dev libtiff-dev \
    && rm -rf /var/lib/apt/lists/*

# Versions / layout
ENV OPENCV_VERSION=4.12.0
WORKDIR /opt

# Fetch OpenCV and opencv_contrib (for aruco)
RUN wget -q https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip -O opencv.zip \
 && unzip -q opencv.zip && rm opencv.zip \
 && mv opencv-${OPENCV_VERSION} opencv
RUN wget -q https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip -O opencv_contrib.zip \
 && unzip -q opencv_contrib.zip && rm opencv_contrib.zip \
 && mv opencv_contrib-${OPENCV_VERSION} opencv_contrib

# Configure and build: include modules GoCV compiles against; GUI backends off
WORKDIR /opt/opencv/build
RUN cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D OPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib/modules \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D BUILD_LIST=core,imgproc,imgcodecs,highgui,videoio,video,photo,calib3d,features2d,objdetect,ml,flann,dnn,aruco \
    -D WITH_QT=OFF -D WITH_GTK=OFF -D WITH_OPENGL=OFF -D WITH_FFMPEG=OFF \
    -D WITH_JPEG=ON -D WITH_PNG=ON -D WITH_TIFF=ON \
    -D WITH_PROTOBUF=ON -D BUILD_PROTOBUF=ON \
    .. \
 && make -j"$(nproc)" \
 && make install

# Go build env to find OpenCV
ENV PKG_CONFIG_PATH=/usr/local/lib/pkgconfig
ENV LD_LIBRARY_PATH=/usr/local/lib

# Build Go app
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=1 go build -ldflags="-s -w" -o /server ./cmd/server

# --- STAGE 2: Runtime ---
FROM debian:bookworm-slim

# Minimal runtime deps for OpenCV core/imgproc/imgcodecs/etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates wget libtbb12 libjpeg62-turbo libpng16-16 libtiff6 \
    libwebp7 libwebpdemux2 libwebpmux3 \
    && rm -rf /var/lib/apt/lists/*

# OpenCV shared libs (copy versioned .so + .412 soname symlinks)
COPY --from=builder /usr/local/lib/libopencv*.so* /usr/local/lib/
RUN ldconfig

WORKDIR /tmp
RUN wget -q https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-1.22.0.tgz -O onnxruntime.tgz \
    && tar -zxf onnxruntime.tgz \
    && ls -l onnxruntime-linux-x64-1.22.0/lib \
    && cp onnxruntime-linux-x64-1.22.0/lib/libonnxruntime.so.1.22.0 /usr/local/lib/ \
    && ln -sf /usr/local/lib/libonnxruntime.so.1.22.0 /usr/local/lib/libonnxruntime.so \
    && ln -sf /usr/local/lib/libonnxruntime.so /usr/local/lib/onnxruntime.so \
    && rm -rf onnxruntime.tgz onnxruntime-linux-x64-1.22.0 \
    && test -f /usr/local/lib/libonnxruntime.so.1.22.0 && test -f /usr/local/lib/libonnxruntime.so
RUN ldconfig

# Ensure loader and wrapper can find ORT
ENV LD_LIBRARY_PATH=/usr/local/lib

# App
WORKDIR /app
COPY --from=builder /server .
COPY models ./models
EXPOSE 8080
ENTRYPOINT ["/app/server"]