import numpy as np
import time
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

def load_csv(path, max_rows=None):
    data = []
    with open(path) as f:
        for line in f:
            if max_rows and len(data) >= max_rows:
                break
            row = []
            for cell in line.strip().split(','):
                try:
                    row.append(float(cell))
                except:
                    break
            if row:
                data.append(row)
    return np.array(data)

def benchmark_kmeans(X, k, name):
    results = {}

    # KMeans
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    t1 = time.perf_counter()
    km.fit(X)
    t2 = time.perf_counter()
    results['kmeans_fit'] = (t2 - t1) * 1000

    t1 = time.perf_counter()
    km.predict(X)
    t2 = time.perf_counter()
    results['kmeans_predict'] = (t2 - t1) * 1000
    results['kmeans_inertia'] = km.inertia_
    results['kmeans_iters'] = km.n_iter_

    # MiniBatchKMeans
    mb = MiniBatchKMeans(n_clusters=k, batch_size=100, n_init=10, random_state=42)
    t1 = time.perf_counter()
    mb.fit(X)
    t2 = time.perf_counter()
    results['minibatch_fit'] = (t2 - t1) * 1000

    t1 = time.perf_counter()
    mb.predict(X)
    t2 = time.perf_counter()
    results['minibatch_predict'] = (t2 - t1) * 1000
    results['minibatch_inertia'] = mb.inertia_

    return results

def benchmark_pca(X, name):
    results = {}
    for n_comp in [2, 5, 10]:
        if n_comp >= X.shape[1]:
            continue
        pca = PCA(n_components=n_comp, random_state=42)
        t1 = time.perf_counter()
        Y = pca.fit_transform(X)
        t2 = time.perf_counter()
        results[f'pca_{n_comp}d'] = {
            'time_ms': (t2 - t1) * 1000,
            'variance': sum(pca.explained_variance_ratio_) * 100
        }
    return results

def print_header():
    print(f"{'Algorithm':<25} {'Fit Time':>12} {'Predict Time':>14} {'Inertia':>12} {'Iters':>8}")
    print("-" * 75)

def print_row(name, fit_ms, predict_ms, inertia, iters="-"):
    print(f"{name:<25} {fit_ms:>10.1f}ms {predict_ms:>12.1f}ms {inertia:>12.0f} {iters:>6}")

def run_benchmarks():
    print("=" * 75)
    print("  SCIKIT-LEARN BENCHMARK")
    print("=" * 75)

    data_dir = "benchmarks/data/"

    datasets = [
        ("Iris", data_dir + "iris.csv", 3),
        ("Wine", data_dir + "wine.csv", 3),
        ("Synthetic 10k", data_dir + "synthetic_10k_32d.csv", 10),
    ]

    all_results = {}

    for name, path, k in datasets:
        print(f"\n=== {name} (k={k}) ===")
        X = load_csv(path)
        print(f"Loaded: {X.shape[0]} samples x {X.shape[1]} features\n")

        print_header()

        results = benchmark_kmeans(X, k, name)

        print_row("sklearn KMeans",
                  results['kmeans_fit'],
                  results['kmeans_predict'],
                  results['kmeans_inertia'],
                  results['kmeans_iters'])

        print_row("sklearn MiniBatch",
                  results['minibatch_fit'],
                  results['minibatch_predict'],
                  results['minibatch_inertia'])

        all_results[name] = results

    # PCA
    print("\n=== PCA: Synthetic 10k ===")
    X = load_csv(data_dir + "synthetic_10k_32d.csv")
    print(f"Input: {X.shape[0]} x {X.shape[1]}")

    pca_results = benchmark_pca(X, "Synthetic")
    for key, val in pca_results.items():
        n = key.split('_')[1]
        print(f"  32d -> {n}: {val['time_ms']:.1f}ms, {val['variance']:.1f}% variance retained")

    return all_results

if __name__ == "__main__":
    run_benchmarks()
