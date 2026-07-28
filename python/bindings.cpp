#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "clustering/clustering.h"

namespace py = pybind11;
using namespace clustering;

py::array_t<float> matrix_to_numpy(const Matrix& m) {
    py::array_t<float> arr({(int)m.rows(), (int)m.cols()});
    auto buf = arr.request();
    float* ptr = static_cast<float*>(buf.ptr);
    for (size_t i = 0; i < m.rows(); ++i)
        for (size_t j = 0; j < m.cols(); ++j)
            ptr[i * m.cols() + j] = m[i][j];
    return arr;
}

py::array_t<float> vector_to_numpy(const Vector& v) {
    py::array_t<float> arr({(int)v.size()});
    auto buf = arr.request();
    float* ptr = static_cast<float*>(buf.ptr);
    for (size_t i = 0; i < v.size(); ++i)
        ptr[i] = v[i];
    return arr;
}

Matrix numpy_to_matrix(py::array_t<float> arr) {
    auto buf = arr.request();
    return Matrix(buf.shape[0], buf.shape[1], static_cast<float*>(buf.ptr));
}

Vector numpy_to_vector(py::array_t<float> arr) {
    auto buf = arr.request();
    return Vector(buf.shape[0], static_cast<float*>(buf.ptr));
}

PYBIND11_MODULE(_clustering, m) {
    m.doc() = "C++ clustering engine";

    py::class_<KMeans>(m, "KMeans")
        .def(py::init<size_t>(), py::arg("k") = 8)
        .def("fit", [](KMeans& self, py::array_t<float> X) {
            self.fit(numpy_to_matrix(X));
        })
        .def("predict", [](KMeans& self, py::array_t<float> X) {
            return vector_to_numpy(self.predict(numpy_to_matrix(X)));
        })
        .def("partial_fit", [](KMeans& self, py::array_t<float> X) {
            self.partial_fit(numpy_to_matrix(X));
        })
        .def_property_readonly("labels", [](KMeans& self) {
            return vector_to_numpy(self.labels());
        })
        .def_property_readonly("centroids", [](KMeans& self) {
            return matrix_to_numpy(self.centroids());
        })
        .def_property_readonly("inertia", &KMeans::inertia)
        .def_property_readonly("n_iter", &KMeans::n_iter);

    py::class_<OnlineKMeans, KMeans>(m, "OnlineKMeans")
        .def(py::init<size_t>(), py::arg("k") = 8)
        .def("partial_fit", [](OnlineKMeans& self, py::array_t<float> X) {
            self.partial_fit(numpy_to_matrix(X));
        })
        .def_property_readonly("points_seen", &OnlineKMeans::points_seen);

    py::class_<MiniBatchKMeans, KMeans>(m, "MiniBatchKMeans")
        .def(py::init<size_t, size_t>(), py::arg("k") = 8, py::arg("batch_size") = 100);

    py::class_<PCA>(m, "PCA")
        .def(py::init<int>(), py::arg("n_components"))
        .def("fit", [](PCA& self, py::array_t<float> X) {
            auto buf = X.request();
            if (buf.ndim != 2) throw std::runtime_error("Expected 2D array");
            self.fit_raw(static_cast<float*>(buf.ptr), buf.shape[0], buf.shape[1]);
        })
        .def("transform", [](PCA& self, py::array_t<float> X) {
            auto buf = X.request();
            if (buf.ndim != 2) throw std::runtime_error("Expected 2D array");
            size_t n = buf.shape[0];
            size_t k = self.n_components();
            py::array_t<float> out({(int)n, (int)k});
            auto obuf = out.request();
            self.transform_raw(static_cast<float*>(buf.ptr), n, static_cast<float*>(obuf.ptr));
            return out;
        })
        .def("fit_transform", [](PCA& self, py::array_t<float> X) {
            auto buf = X.request();
            if (buf.ndim != 2) throw std::runtime_error("Expected 2D array");
            size_t n = buf.shape[0];
            size_t k = self.n_components();
            py::array_t<float> out({(int)n, (int)k});
            auto obuf = out.request();
            self.fit_transform_raw(static_cast<float*>(buf.ptr), n, buf.shape[1], static_cast<float*>(obuf.ptr));
            return out;
        })
        .def("inverse_transform", [](PCA& self, py::array_t<float> X) {
            auto buf = X.request();
            if (buf.ndim != 2) throw std::runtime_error("Expected 2D array");
            size_t n = buf.shape[0];
            size_t d = self.components().rows() > 0 ? self.components().cols() : 0;
            py::array_t<float> out({(int)n, (int)d});
            auto obuf = out.request();
            self.inverse_transform_raw(static_cast<float*>(buf.ptr), n, static_cast<float*>(obuf.ptr));
            return out;
        })
        .def_property_readonly("components", [](PCA& self) {
            return matrix_to_numpy(self.components());
        })
        .def_property_readonly("explained_variance_ratio", [](PCA& self) {
            return vector_to_numpy(self.explained_variance_ratio());
        })
        .def_property_readonly("total_explained_variance_ratio", &PCA::total_explained_variance_ratio);

    py::class_<TSNEConfig>(m, "TSNEConfig")
        .def(py::init<>())
        .def_readwrite("n_components", &TSNEConfig::n_components)
        .def_readwrite("perplexity", &TSNEConfig::perplexity)
        .def_readwrite("learning_rate", &TSNEConfig::learning_rate)
        .def_readwrite("n_iter", &TSNEConfig::n_iter);

    py::class_<TSNE>(m, "TSNE")
        .def(py::init<TSNEConfig>(), py::arg("config") = TSNEConfig())
        .def("fit", [](TSNE& self, py::array_t<float> X) {
            self.fit(numpy_to_matrix(X));
        })
        .def("fit_transform", [](TSNE& self, py::array_t<float> X) {
            return matrix_to_numpy(self.fit_transform(numpy_to_matrix(X)));
        })
        .def_property_readonly("embedding", [](TSNE& self) {
            return matrix_to_numpy(self.embedding());
        })
        .def_property_readonly("kl_divergence", &TSNE::kl_divergence);

    py::class_<DriftMetrics>(m, "DriftMetrics")
        .def_readonly("silhouette_score", &DriftMetrics::silhouette_score)
        .def_readonly("davies_bouldin_index", &DriftMetrics::davies_bouldin_index)
        .def_readonly("calinski_harabasz_score", &DriftMetrics::calinski_harabasz_score)
        .def_readonly("cluster_stability", &DriftMetrics::cluster_stability)
        .def_readonly("drift_detected", &DriftMetrics::drift_detected);

    py::class_<DriftDetector>(m, "DriftDetector")
        .def(py::init<>())
        .def("set_threshold", &DriftDetector::set_threshold)
        .def("set_window_size", &DriftDetector::set_window_size)
        .def("check", [](DriftDetector& self, py::array_t<float> X,
                         py::array_t<float> labels, py::array_t<float> centroids) {
            return self.check(numpy_to_matrix(X), numpy_to_vector(labels),
                            numpy_to_matrix(centroids));
        })
        .def_property_readonly("is_drifting", &DriftDetector::is_drifting);

    py::class_<VersionManager>(m, "VersionManager")
        .def(py::init<>())
        .def("set_storage_path", &VersionManager::set_storage_path)
        .def("save", [](VersionManager& self, py::array_t<float> c, py::array_t<float> l) {
            return self.save(numpy_to_matrix(c), numpy_to_vector(l));
        })
        .def("load", [](VersionManager& self, size_t id) {
            Matrix c; Vector l;
            bool ok = self.load(id, c, l);
            if (!ok) throw std::runtime_error("Version not found");
            return py::make_tuple(matrix_to_numpy(c), vector_to_numpy(l));
        })
        .def("list_versions", &VersionManager::list_versions)
        .def("has_versions", &VersionManager::has_versions);

    py::class_<FeatureStore>(m, "FeatureStore")
        .def(py::init<>())
        .def("set_cache_path", &FeatureStore::set_cache_path)
        .def("preprocess", [](FeatureStore& self, py::array_t<float> X, std::string op) {
            return matrix_to_numpy(self.preprocess(numpy_to_matrix(X), op));
        });

    py::class_<OnlineConfig>(m, "OnlineConfig")
        .def(py::init<>())
        .def_readwrite("k", &OnlineConfig::k)
        .def_readwrite("window_size", &OnlineConfig::window_size)
        .def_readwrite("forgetting_factor", &OnlineConfig::forgetting_factor)
        .def_readwrite("auto_retrain", &OnlineConfig::auto_retrain)
        .def_readwrite("drift_threshold", &OnlineConfig::drift_threshold);

    py::class_<KMeansConfig>(m, "KMeansConfig")
        .def(py::init<>())
        .def_readwrite("k", &KMeansConfig::k)
        .def_readwrite("max_iter", &KMeansConfig::max_iter)
        .def_readwrite("tol", &KMeansConfig::tol)
        .def_readwrite("max_threads", &KMeansConfig::max_threads);

    m.def("l2_distance", [](py::array_t<float> a, py::array_t<float> b) {
        auto ba = a.request(), bb = b.request();
        if (ba.shape[0] != bb.shape[0]) throw std::runtime_error("Dim mismatch");
        return l2_distance_avx2(static_cast<float*>(ba.ptr),
                               static_cast<float*>(bb.ptr), ba.shape[0]);
    }, "L2 distance between vectors");

    m.def("pca_fit_transform", [](py::array_t<float> X, int n_components) {
        auto buf = X.request();
        if (buf.ndim != 2) throw std::runtime_error("Expected 2D array");
        size_t n = buf.shape[0];
        float* data = static_cast<float*>(buf.ptr);
        PCA pca(n_components);
        pca.fit_raw(data, n, buf.shape[1]);
        py::array_t<float> out({(int)n, n_components});
        auto obuf = out.request();
        pca.transform_raw(data, n, static_cast<float*>(obuf.ptr));
        return py::make_tuple(out, pca.components(), pca.explained_variance_ratio());
    }, py::arg("X"), py::arg("n_components"));
}
