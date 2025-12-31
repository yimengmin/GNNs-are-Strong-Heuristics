#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <limits>
#include <cmath>
#include <stdexcept>

namespace py = pybind11;

static inline double dist2d(double x1, double y1, double x2, double y2) {
    const double dx = x1 - x2;
    const double dy = y1 - y2;
    return std::sqrt(dx * dx + dy * dy);
}

// Nearest Neighbor greedy tour for Euclidean TSP
// coords: (N,2) numpy array (float32/float64)
// start: starting node index
py::tuple greedy_tsp_nn(py::array coords, int start = 0) {
    py::buffer_info buf = coords.request();
    if (buf.ndim != 2 || buf.shape[1] != 2) {
        throw std::runtime_error("coords must have shape (N,2)");
    }
    const int64_t n = buf.shape[0];
    if (n <= 1) {
        // trivial
        py::array_t<int64_t> tour_arr({n});
        auto t = tour_arr.mutable_unchecked<1>();
        if (n == 1) t(0) = 0;
        return py::make_tuple(tour_arr, 0.0);
    }
    if (start < 0 || start >= (int)n) {
        throw std::runtime_error("start index out of range");
    }

    // Read coords as double regardless of dtype
    // Support float32/float64 input.
    const bool is_f64 = (buf.format == py::format_descriptor<double>::format());
    const bool is_f32 = (buf.format == py::format_descriptor<float>::format());
    if (!is_f64 && !is_f32) {
        throw std::runtime_error("coords dtype must be float32 or float64");
    }

    auto get_xy = [&](int64_t i, double &x, double &y) {
        if (is_f64) {
            const auto *p = static_cast<const double*>(buf.ptr) + i * 2;
            x = p[0]; y = p[1];
        } else {
            const auto *p = static_cast<const float*>(buf.ptr) + i * 2;
            x = (double)p[0]; y = (double)p[1];
        }
    };

    std::vector<int64_t> tour;
    tour.reserve(n);
    std::vector<char> used(n, 0);

    int64_t cur = start;
    used[cur] = 1;
    tour.push_back(cur);

    double total = 0.0;

    for (int64_t step = 1; step < n; ++step) {
        double cx, cy;
        get_xy(cur, cx, cy);

        int64_t best = -1;
        double best_d = std::numeric_limits<double>::infinity();

        for (int64_t j = 0; j < n; ++j) {
            if (used[j]) continue;
            double x, y;
            get_xy(j, x, y);
            const double d = dist2d(cx, cy, x, y);
            if (d < best_d) {
                best_d = d;
                best = j;
            }
        }

        // move
        used[best] = 1;
        tour.push_back(best);
        total += best_d;
        cur = best;
    }

    // close the tour
    double sx, sy, lx, ly;
    get_xy(tour.front(), sx, sy);
    get_xy(tour.back(),  lx, ly);
    total += dist2d(lx, ly, sx, sy);

    py::array_t<int64_t> tour_arr({n});
    auto out = tour_arr.mutable_unchecked<1>();
    for (int64_t i = 0; i < n; ++i) out(i) = tour[i];

    return py::make_tuple(tour_arr, total);
}

PYBIND11_MODULE(greedy_tsp, m) {
    m.doc() = "Greedy (Nearest Neighbor) TSP baseline (C++/pybind11)";
    m.def("greedy_tsp_nn", &greedy_tsp_nn, py::arg("coords"), py::arg("start") = 0,
          "Compute a greedy NN TSP tour. Returns (tour, length).");
}

