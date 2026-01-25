#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cmath>
#include <vector>
#include <random>
#include <limits>

namespace py = pybind11;

struct Point {
    double x;
    double y;
};

double euclidean_distance(const Point& a, const Point& b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    return std::sqrt(dx*dx + dy*dy);
}

double tour_length(const std::vector<Point>& coords,
                   const std::vector<int>& tour) {
    double total = 0.0;
    int n = tour.size();
    for (int i = 0; i < n; i++) {
        const Point& p1 = coords[tour[i]];
        const Point& p2 = coords[tour[(i + 1) % n]];
        total += euclidean_distance(p1, p2);
    }
    return total;
}

std::vector<int> nearest_neighbor_tour(const std::vector<Point>& coords) {
    int n = coords.size();
    std::vector<int> tour;
    tour.reserve(n);

    std::vector<bool> visited(n, false);
    int current = 0;
    tour.push_back(current);
    visited[current] = true;

    for (int s = 1; s < n; s++) {
        int best = -1;
        double best_dist = std::numeric_limits<double>::infinity();
        for (int j = 0; j < n; j++) {
            if (!visited[j]) {
                double d = euclidean_distance(coords[current], coords[j]);
                if (d < best_dist) {
                    best_dist = d;
                    best = j;
                }
            }
        }
        tour.push_back(best);
        visited[best] = true;
        current = best;
    }

    return tour;
}

enum class InitMode { Random, NearestNeighbor };

py::tuple two_opt_cpp(py::array_t<double> coords_np,
                      std::string mode) {

    auto buf = coords_np.request();
    int n = buf.shape[0];

    const double* ptr = static_cast<double*>(buf.ptr);

    std::vector<Point> coords(n);
    for (int i = 0; i < n; i++) {
        coords[i].x = ptr[2*i];
        coords[i].y = ptr[2*i + 1];
    }

    std::vector<int> tour(n);
    if (mode == "random") {
        for (int i = 0; i < n; i++) tour[i] = i;
        static std::mt19937 rng(std::random_device{}());
        std::shuffle(tour.begin(), tour.end(), rng);
    }
    else if (mode == "nn") {
        tour = nearest_neighbor_tour(coords);
    }
    else {
        throw std::runtime_error("Unknown mode: " + mode);
    }

    double best_len = tour_length(coords, tour);

    bool improved = true;
    while (improved) {
        improved = false;

        for (int i = 1; i < n-2; i++) {
            for (int k = i+1; k < n-1; k++) {

                int a = tour[i-1];
                int b = tour[i];
                int c = tour[k];
                int d = tour[(k+1) % n];

                double old_dist =
                    euclidean_distance(coords[a], coords[b]) +
                    euclidean_distance(coords[c], coords[d]);

                double new_dist =
                    euclidean_distance(coords[a], coords[c]) +
                    euclidean_distance(coords[b], coords[d]);

                if (new_dist + 1e-12 < old_dist) {
                    std::reverse(tour.begin()+i, tour.begin()+k+1);
                    best_len += (new_dist - old_dist);
                    improved = true;
                }
            }
        }
    }

    return py::make_tuple(tour, best_len);
}

PYBIND11_MODULE(two_opt_cpp, m) {
    m.doc() = "C++ 2-opt module";
    m.def("two_opt", &two_opt_cpp,
          "TSP 2-opt heuristic",
          py::arg("coords_np"),
          py::arg("mode") = "random");
}

