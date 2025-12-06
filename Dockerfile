FROM ubuntu:22.04

# Install tools and compilers
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY . .

# Configure, build, and run tests
RUN cmake -S . -B build && \
    cmake --build build && \
    cd build && ctest --output-on-failure

# Default behavior: run logistic regression example
ENTRYPOINT ["./build/train_logistic"]
CMD ["examples/data/logistic_toy.csv"]
