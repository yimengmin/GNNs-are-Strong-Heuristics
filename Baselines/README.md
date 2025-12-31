# Greedy TSP Baseline (C++ / pybind11)

This repository provides a **C++ implementation of a greedy (Nearest Neighbor) TSP heuristic** with a Python wrapper via **pybind11**, together with an evaluation script that reports **per-instance cost and runtime statistics** on TSP datasets stored in `.pt` format.

---

## 1. Requirements

### System

* Linux or macOS
* Python ≥ 3.8
* A C++17-compatible compiler (`g++`, `clang++`)

### Python packages

```bash
pip install numpy torch pybind11 setuptools wheel
```

---

## 2. File Structure

```text
.
├── greedy_tsp.cpp          # C++ greedy (Nearest Neighbor) TSP
├── run_greedy.py           # build script (setuptools + pybind11)
├── run_eval_greedy.py      # evaluation on .pt TSP datasets
├── README.md
```

The dataset file must contain a dictionary with key:

```python
"coordinates": Tensor of shape (B, N, 2)
```

---

## 3. Build the C++ Extension

Compile the C++ greedy solver into a Python extension **in-place**:

```bash
python run_greedy.py build_ext --inplace
```

If successful, this will generate a shared library such as:

* `greedy_tsp.cpython-3xx-*.so` (Linux/macOS)
* `greedy_tsp.pyd` (Windows)

in the current directory.

---

## 4. Run the Greedy Baseline

Evaluate the greedy TSP solver on a dataset:

```bash
python run_eval_greedy.py
```

By default, the script loads:

```python
dataset_path = "../data/tsp_500_uniform_test.pt"
```

You can change this path directly inside `run_eval_greedy.py`.

---

## 5. Reported Metrics

For each TSP instance, the script reports:

* **Cost per instance**
* **Runtime per instance**

After processing all instances, it reports:

* **Mean tour cost**
* **Standard deviation of tour cost**
* **Mean runtime per instance**
* **Standard deviation of runtime**

Runtime is measured using `time.perf_counter()` and includes **only the greedy algorithm**, excluding data loading.

Example output:

```text
[  1/1000] cost=21.374821  time=0.812 ms
...
============================================================
Mean cost     : 23.108472
Std  cost     : 1.742913
Mean runtime  : 0.791 ms / instance
Std  runtime  : 0.064 ms / instance
============================================================
```

---

## 6. Optional: Quick Testing on a Subset

To evaluate only the first `K` instances (useful for debugging), edit:

```python
max_instances = 100
```

inside `run_eval_greedy.py`.

---

## 7. Notes on Reproducibility

* The greedy heuristic is **deterministic** given a fixed starting node (`start=0` by default).
* Input coordinates are converted to `float64` for stable distance computation.
* The reported runtime reflects **single-threaded CPU execution**.

---

## 8. Common Issues

### Compiler not found

Ensure a C++ compiler is available:

```bash
g++ --version
```

### pybind11 not found

```bash
pip install pybind11
```

### Dataset format error

The `.pt` file must contain:

```python
{"coordinates": Tensor(B, N, 2)}
```


