# cpp-ml-mini-lib

![CI](https://github.com/Ghada1997/cpp-ml-mini-lib/actions/workflows/ci.yml/badge.svg)
![Docker Image](https://img.shields.io/badge/docker%20image-ghcr.io%2FGhada1997%2Fcpp--ml--mini--lib-blue)
![C++17](https://img.shields.io/badge/c%2B%2B-17-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

A lightweight **C++17 machine learning library**.

It demonstrates how classical ML algorithms can be implemented from scratch using modern C++ practices. The goal is to explore how core ML algorithms work *under the hood* and to demonstrate modern C++ engineering practices (modular design, tests, CI).

### The library includes:

- **Logistic Regression** (binary classification via gradient descent)
- **Multinomial Naive Bayes** (count/frequency-based classification)
- **K-Means** (unsupervised clustering)
- **CSV dataset loader** (supervised + unsupervised formats)
- **Evaluation metrics** (Accuracy, F1-score)
- **CLI demo applications**
- **Unit tests** using GoogleTest
- **CI/CD pipeline** (automated build, test, Docker build & deployment)

---

## 📦 Project Structure
```text
cpp-ml-mini-lib/
├── include/ml/ # Public headers
├── src/ # Algorithm implementations
├── tests/ # GoogleTest unit tests
├── apps/ # Command-line demo applications
├── examples/data/ # Tiny toy datasets
├── .github/workflows/ # CI/CD (GitHub Actions + Docker deployment)
├── Dockerfile # Docker build definition
└── CMakeLists.txt # Build configuration
```

---

## 🛠 Build Instructions

### Requirements:
- **CMake ≥ 3.16**
- **C++17 compiler**
- **git**

```bash
git clone https://github.com/Ghada1997/cpp-ml-mini-lib.git
cd cpp-ml-mini-lib

mkdir build
cd build
cmake ..
cmake --build .
```
### 🧪 Run Unit Tests
```bash
ctest --output-on-failure
```

### ▶️ Run Example Applications
Logistic Regression
```bash
./train_logistic ../examples/data/logistic_toy.csv
```
Naive Bayes
```bash
./train_naive_bayes ../examples/data/naive_bayes_toy.csv
```
K-Means
```bash
./run_kmeans ../examples/data/kmeans_toy.csv 
```
--- 

## 🚀 Continuous Integration & Deployment (CI/CD)

This repository includes a complete **CI/CD pipeline** using GitHub Actions.

### ✔ Automated Build  
Every push or pull request triggers:

- CMake configuration  
- Full compilation of the library and CLI applications  

### ✔ Automated Testing  
All GoogleTest unit tests run automatically to validate:

- Logistic Regression  
- Naive Bayes  
- K-Means  
- CSV Loader  
- Metrics  

This ensures algorithm correctness on every commit.

### ✔ Docker Image Build  
The CI pipeline builds a Docker image that contains:

- The compiled C++ ML library  
- All example CLI applications  
- A default entrypoint to run the logistic regression demo  

### ✔ Docker Deployment to GitHub Container Registry (GHCR)

> **Note:** Replace `<your-username>` with your GitHub username when running these commands.

The Docker image is automatically published to:
```bash
ghcr.io/<your-username>/cpp-ml-mini-lib:latest
```

This provides reproducible builds across environments and simplifies usage without local compilation.

---

## 🐳 Using the Docker Image

### Pull the image
```bash
docker pull ghcr.io/<your-username>/cpp-ml-mini-lib:latest
```

### Run it
```bash
docker run --rm ghcr.io/<your-username>/cpp-ml-mini-lib:latest
``` 
Default entrypoint runs the logistic regression example.

---

## 🌱 Future Improvements

- Add real datasets (Iris, simple text examples)
- Add additional algorithms (Linear Regression, PCA, SVM)
- Add benchmarking scripts
- Add Python bindings via pybind11
- Add visualizations for clustering/classification

---

## 📄 License

This project is released under the MIT License.

---