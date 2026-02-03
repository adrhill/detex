window.BENCHMARK_DATA = {
  "lastUpdate": 1770119699685,
  "repoUrl": "https://github.com/adrhill/detex",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "Adrian Hill",
            "username": "adrhill"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "aa3ca4f0cca7a93af150537d802fc1b5d46f668b",
          "message": "Add code coverage and benchmark tracking to CI (#5)\n\n* Add code coverage and benchmark tracking to CI\n\n- Add pytest-cov for coverage reporting, upload to Codecov\n- Add Benchmarks job using github-action-benchmark to track\n  performance regressions and comment on PRs\n\nCo-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>\n\n* Split workflow\n\n---------\n\nCo-authored-by: Claude Opus 4.5 <noreply@anthropic.com>",
          "timestamp": "2026-02-03T12:54:33+01:00",
          "tree_id": "b60e8e3275205f593c72a13358ed96332f273f6c",
          "url": "https://github.com/adrhill/detex/commit/aa3ca4f0cca7a93af150537d802fc1b5d46f668b"
        },
        "date": 1770119698947,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_bench_diagonal_n100",
            "value": 1466.2904947242746,
            "unit": "iter/sec",
            "range": "stddev: 0.00004935802412810461",
            "extra": "mean: 681.9931000016766 usec\nrounds: 10"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_diagonal_n500",
            "value": 557.173469844974,
            "unit": "iter/sec",
            "range": "stddev: 0.004499776162785215",
            "extra": "mean: 1.7947731794879547 msec\nrounds: 39"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_diagonal_n1000",
            "value": 398.13579410297245,
            "unit": "iter/sec",
            "range": "stddev: 0.005707845923022922",
            "extra": "mean: 2.511705842106132 msec\nrounds: 38"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_dense_sum_n100",
            "value": 1503.6538361922194,
            "unit": "iter/sec",
            "range": "stddev: 0.00008279921712267961",
            "extra": "mean: 665.0466855671726 usec\nrounds: 582"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_dense_sum_n500",
            "value": 992.8296237138874,
            "unit": "iter/sec",
            "range": "stddev: 0.0016643088178266493",
            "extra": "mean: 1.0072221619045676 msec\nrounds: 525"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_dense_matmul_n100",
            "value": 132.71147173004096,
            "unit": "iter/sec",
            "range": "stddev: 0.010248879774024246",
            "extra": "mean: 7.5351436237115985 msec\nrounds: 194"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_mlp_layer",
            "value": 97.92185282559701,
            "unit": "iter/sec",
            "range": "stddev: 0.01265652979526441",
            "extra": "mean: 10.212225066666605 msec\nrounds: 150"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_elementwise_chain",
            "value": 897.3785275048924,
            "unit": "iter/sec",
            "range": "stddev: 0.000029157812275631677",
            "extra": "mean: 1.1143569512192815 msec\nrounds: 41"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_mixed_ops",
            "value": 1478.212212538373,
            "unit": "iter/sec",
            "range": "stddev: 0.00002216304332103185",
            "extra": "mean: 676.492855029799 usec\nrounds: 338"
          }
        ]
      }
    ]
  }
}