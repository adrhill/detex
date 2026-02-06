window.BENCHMARK_DATA = {
  "lastUpdate": 1770394107008,
  "repoUrl": "https://github.com/adrhill/asdex",
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
      },
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
          "id": "c02a100d759b4c8a1c6099bc09cc55ca74aa92ef",
          "message": "Add SymPy-based randomized tests (#6)\n\n* Add SymPy-based randomized tests for Jacobian sparsity\n\nGenerate random mathematical expressions using SymPy primitives and\ncompare detex's sparsity detection against symbolic differentiation\nground truth. Tests verify detex doesn't miss any dependencies.\n\n- Add sympy dev dependency\n- Add SympyToJax converter using dictionary-based dispatch\n- Add TestSympyComparison with randomized expression tests\n- Add TestSympyEdgeCases for specific patterns (nested, polynomial, etc.)\n\nCo-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>\n\n* Test for exact patterns\n\n---------\n\nCo-authored-by: Claude Opus 4.5 <noreply@anthropic.com>",
          "timestamp": "2026-02-03T14:48:06+01:00",
          "tree_id": "55fb75a7062fee1390510a04694e38f1c3c61cf3",
          "url": "https://github.com/adrhill/detex/commit/c02a100d759b4c8a1c6099bc09cc55ca74aa92ef"
        },
        "date": 1770126511519,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_bench_diagonal_n100",
            "value": 1462.3523956205077,
            "unit": "iter/sec",
            "range": "stddev: 0.00005041444558289315",
            "extra": "mean: 683.8297000058446 usec\nrounds: 10"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_diagonal_n500",
            "value": 567.0589612816366,
            "unit": "iter/sec",
            "range": "stddev: 0.0044292601948708625",
            "extra": "mean: 1.763485048785497 msec\nrounds: 41"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_diagonal_n1000",
            "value": 407.24900140915554,
            "unit": "iter/sec",
            "range": "stddev: 0.005480715921880039",
            "extra": "mean: 2.455500189171289 msec\nrounds: 37"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_dense_sum_n100",
            "value": 1614.2324042045805,
            "unit": "iter/sec",
            "range": "stddev: 0.00002203174566286637",
            "extra": "mean: 619.4894845347587 usec\nrounds: 582"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_dense_sum_n500",
            "value": 1026.298318097376,
            "unit": "iter/sec",
            "range": "stddev: 0.0012895637625450827",
            "extra": "mean: 974.3755615364062 usec\nrounds: 520"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_dense_matmul_n100",
            "value": 134.45649018074488,
            "unit": "iter/sec",
            "range": "stddev: 0.010622471106826242",
            "extra": "mean: 7.437350169231229 msec\nrounds: 195"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_mlp_layer",
            "value": 105.00214557028332,
            "unit": "iter/sec",
            "range": "stddev: 0.01198516860705706",
            "extra": "mean: 9.523614918236587 msec\nrounds: 159"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_elementwise_chain",
            "value": 445.7049475553997,
            "unit": "iter/sec",
            "range": "stddev: 0.0068765923204995956",
            "extra": "mean: 2.2436367500176857 msec\nrounds: 40"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_mixed_ops",
            "value": 1484.0313588207578,
            "unit": "iter/sec",
            "range": "stddev: 0.000022095700115448225",
            "extra": "mean: 673.8402083326735 usec\nrounds: 336"
          }
        ]
      },
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
          "id": "f0f3fca2fce80769bcc98b66a33b9dcd3cc35ef4",
          "message": "Increase code coverage, update CI (#7)\n\n* Remove index set abstraction\n\nCo-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>\n\n* Increase code coverage\n\nCo-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>\n\n* Add TODOs\n\nCo-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>\n\n* Update workflows\n\nCo-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>\n\n* Only test on Ubuntu\n\n---------\n\nCo-authored-by: Claude Opus 4.5 <noreply@anthropic.com>",
          "timestamp": "2026-02-03T15:18:17+01:00",
          "tree_id": "b45e02f69ae9673f8cc4c3b45c8a0b1ef11cdcf6",
          "url": "https://github.com/adrhill/detex/commit/f0f3fca2fce80769bcc98b66a33b9dcd3cc35ef4"
        },
        "date": 1770128355266,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_bench_diagonal_n100",
            "value": 1537.8648469013847,
            "unit": "iter/sec",
            "range": "stddev: 0.000051975740706799664",
            "extra": "mean: 650.2521999998123 usec\nrounds: 10"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_diagonal_n500",
            "value": 606.6322497227576,
            "unit": "iter/sec",
            "range": "stddev: 0.0050500295740285184",
            "extra": "mean: 1.6484451666673159 msec\nrounds: 42"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_diagonal_n1000",
            "value": 917.6842537313044,
            "unit": "iter/sec",
            "range": "stddev: 0.000019268614970056113",
            "extra": "mean: 1.0896994210525024 msec\nrounds: 38"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_dense_sum_n100",
            "value": 1676.344485118207,
            "unit": "iter/sec",
            "range": "stddev: 0.000024975786507968366",
            "extra": "mean: 596.5360991595264 usec\nrounds: 595"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_dense_sum_n500",
            "value": 1296.684871317438,
            "unit": "iter/sec",
            "range": "stddev: 0.00005895894304937807",
            "extra": "mean: 771.1973989362544 usec\nrounds: 564"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_dense_matmul_n100",
            "value": 243.65972628953114,
            "unit": "iter/sec",
            "range": "stddev: 0.006932237806888015",
            "extra": "mean: 4.104084065216998 msec\nrounds: 276"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_mlp_layer",
            "value": 197.84410877455878,
            "unit": "iter/sec",
            "range": "stddev: 0.007662972382939562",
            "extra": "mean: 5.05448459493676 msec\nrounds: 237"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_elementwise_chain",
            "value": 1255.6461199664284,
            "unit": "iter/sec",
            "range": "stddev: 0.00003106920558504024",
            "extra": "mean: 796.4027317081476 usec\nrounds: 41"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_mixed_ops",
            "value": 1614.5660259488675,
            "unit": "iter/sec",
            "range": "stddev: 0.000016962861398074534",
            "extra": "mean: 619.3614779007306 usec\nrounds: 362"
          }
        ]
      },
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
          "id": "bfef6ac65826c05ba0865f316f72b2643e9e40e1",
          "message": "Add edge case tests for sparsity detection (#8)\n\nAdd 18 new tests documenting conservative fallback behavior and one bug:\n- transpose, matmul, argmax, gather, stack, reverse, pad, tile, split,\n  scatter, iota/eye, reduce_max, sort, where_mask, reduce_along_axis\n- Bug: empty array concatenate causes index out-of-bounds error\n- roll and nested_slice_concat work precisely (no fallback needed)\n\nUpdate TODO.md with new primitive coverage items and test references.\n\nCo-authored-by: Claude Opus 4.5 <noreply@anthropic.com>",
          "timestamp": "2026-02-03T15:51:06+01:00",
          "tree_id": "fc5438f246cc7be4d67ef6fd7ab8694c90b611ca",
          "url": "https://github.com/adrhill/detex/commit/bfef6ac65826c05ba0865f316f72b2643e9e40e1"
        },
        "date": 1770130943830,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_bench_diagonal_n100",
            "value": 1536.3371387379527,
            "unit": "iter/sec",
            "range": "stddev: 0.00004958445079084393",
            "extra": "mean: 650.8988000000215 usec\nrounds: 10"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_diagonal_n500",
            "value": 602.6687952128033,
            "unit": "iter/sec",
            "range": "stddev: 0.004909851353180239",
            "extra": "mean: 1.6592861749991528 msec\nrounds: 40"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_diagonal_n1000",
            "value": 904.1320005348431,
            "unit": "iter/sec",
            "range": "stddev: 0.00002889549831159516",
            "extra": "mean: 1.1060331891896822 msec\nrounds: 37"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_dense_sum_n100",
            "value": 1669.5059522413458,
            "unit": "iter/sec",
            "range": "stddev: 0.000045144833579399625",
            "extra": "mean: 598.9795955249393 usec\nrounds: 581"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_dense_sum_n500",
            "value": 1250.7586018433287,
            "unit": "iter/sec",
            "range": "stddev: 0.00009220179592936867",
            "extra": "mean: 799.5147892856634 usec\nrounds: 560"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_dense_matmul_n100",
            "value": 239.0532684922411,
            "unit": "iter/sec",
            "range": "stddev: 0.0074242391169871925",
            "extra": "mean: 4.1831680708957 msec\nrounds: 268"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_mlp_layer",
            "value": 187.2420263439154,
            "unit": "iter/sec",
            "range": "stddev: 0.008856010814062612",
            "extra": "mean: 5.34068136051496 msec\nrounds: 233"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_elementwise_chain",
            "value": 1239.0740222820748,
            "unit": "iter/sec",
            "range": "stddev: 0.000028043625561342584",
            "extra": "mean: 807.0542857142963 usec\nrounds: 42"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_mixed_ops",
            "value": 1609.0334478925013,
            "unit": "iter/sec",
            "range": "stddev: 0.00001633380239054108",
            "extra": "mean: 621.4911202186579 usec\nrounds: 366"
          }
        ]
      },
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
          "id": "cd763819109ea01fb6ff6a34689b0f7797af1dd4",
          "message": "Improve `vmap` support (#9)\n\n* Add precise multi-dimensional index tracking for vmap support\n\nVmapped functions now correctly produce block-diagonal Jacobian sparsity\npatterns instead of falling back to dense (conservative) patterns.\n\nChanges to primitive handlers in _propagate.py:\n- Add _compute_strides helper for flat↔coordinate conversion\n- _propagate_slice: handle multi-dimensional slices precisely\n- _propagate_broadcast_in_dim: handle size-1 dimension broadcasting\n- _propagate_concatenate: handle concatenation along inner dimensions\n- _propagate_reduce_sum: support partial reduction with axes parameter\n\nCo-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>\n\n* Split test_jacobian_sparsity.py into focused test modules\n\n- test_jacobian_sparsity.py: Core API + element-wise operations\n- test_array_ops.py: Array manipulation (slice, concat, broadcast, etc.)\n- test_control_flow.py: Conditionals (where, select)\n- test_reductions.py: Reduction operations\n- test_vmap.py: Batched/vmapped operations\n\nAdd pytest markers (elementwise, array_ops, control_flow, reduction, vmap,\nfallback, bug) for selective test runs. Add tests/CLAUDE.md documenting\nthe test structure and conventions.\n\nCo-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>\n\n* Raise informative error for missing jaxpr parameter\n\nReplace silent fallback with a ValueError that includes the primitive\nname and links to the issue tracker. This achieves 100% test coverage.\n\nCo-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>\n\n* Raise error for unknown primitives instead of silent fallback\n\n- Rename _propagate_default to _propagate_conservative_fallback\n- Add _propagate_throw_error for unknown primitives\n- Explicitly list primitives that use conservative fallback with TODO\n\nCo-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.5 <noreply@anthropic.com>",
          "timestamp": "2026-02-04T23:42:59+01:00",
          "tree_id": "f2da0ccfb011af0abeb5522504a9c5896220b0d9",
          "url": "https://github.com/adrhill/detex/commit/cd763819109ea01fb6ff6a34689b0f7797af1dd4"
        },
        "date": 1770245002133,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_bench_diagonal_n100",
            "value": 1575.3047663364048,
            "unit": "iter/sec",
            "range": "stddev: 0.000047490307958336215",
            "extra": "mean: 634.7978000000865 usec\nrounds: 10"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_diagonal_n500",
            "value": 688.4146018245003,
            "unit": "iter/sec",
            "range": "stddev: 0.003942147431087655",
            "extra": "mean: 1.4526129999998652 msec\nrounds: 43"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_diagonal_n1000",
            "value": 933.1704826417675,
            "unit": "iter/sec",
            "range": "stddev: 0.00002169609146217004",
            "extra": "mean: 1.0716155500000824 msec\nrounds: 40"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_dense_sum_n100",
            "value": 1718.0396519622934,
            "unit": "iter/sec",
            "range": "stddev: 0.000019814019198566518",
            "extra": "mean: 582.0587428572036 usec\nrounds: 630"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_dense_sum_n500",
            "value": 1352.3726741416901,
            "unit": "iter/sec",
            "range": "stddev: 0.000017066064393151597",
            "extra": "mean: 739.4411460100446 usec\nrounds: 589"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_dense_matmul_n100",
            "value": 279.2694816448513,
            "unit": "iter/sec",
            "range": "stddev: 0.004917649239072918",
            "extra": "mean: 3.5807707813620184 msec\nrounds: 279"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_mlp_layer",
            "value": 238.32036429459976,
            "unit": "iter/sec",
            "range": "stddev: 0.004618820795154789",
            "extra": "mean: 4.196032525209847 msec\nrounds: 238"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_elementwise_chain",
            "value": 1269.443913747097,
            "unit": "iter/sec",
            "range": "stddev: 0.000021071632379682943",
            "extra": "mean: 787.7464999995451 usec\nrounds: 44"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_mixed_ops",
            "value": 1438.522117324531,
            "unit": "iter/sec",
            "range": "stddev: 0.000016027268864223224",
            "extra": "mean: 695.1578901406629 usec\nrounds: 355"
          }
        ]
      },
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
          "id": "56aa24a7c46166572b776bc73e8ac1670408b77f",
          "message": "Refactor and document propagation rules (#10)\n\n* Rename internals\n\n* Further simplify names\n\n* Improve docstrings for propagation rules\n\nEach rule now documents:\n- Summary of the operation\n- Mathematical explanation\n- Concrete example with input/output deps\n- Jaxpr parameters\n\nCo-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>\n\n* Refactor _propagate.py into module folder\n\nSplit into focused submodules:\n- _commons.py: types, constants, utilities\n- _elementwise.py: element-wise operations\n- _indexing.py: shape manipulation operations\n- _reduction.py: reduction operations\n- _conv.py: convolution\n\nCo-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>\n\n* Reorganize tests to mirror source structure\n\nMove propagation-related tests into tests/_propagate/:\n- test_array_ops.py → test_indexing.py\n- test_reductions.py → test_reduction.py\n- test_conv.py → test_conv.py\n- test_propagate_internals.py → test_internals.py\n\nCo-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.5 <noreply@anthropic.com>",
          "timestamp": "2026-02-05T01:58:23+01:00",
          "tree_id": "e1cffc960b0e091a4b1256da7399db8c1c288212",
          "url": "https://github.com/adrhill/detex/commit/56aa24a7c46166572b776bc73e8ac1670408b77f"
        },
        "date": 1770253125600,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_bench_diagonal_n100",
            "value": 1565.8658776351906,
            "unit": "iter/sec",
            "range": "stddev: 0.00006931991717497248",
            "extra": "mean: 638.6243000008562 usec\nrounds: 10"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_diagonal_n500",
            "value": 656.2551309920754,
            "unit": "iter/sec",
            "range": "stddev: 0.004356937674137892",
            "extra": "mean: 1.5237976097623465 msec\nrounds: 41"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_diagonal_n1000",
            "value": 918.3901731462745,
            "unit": "iter/sec",
            "range": "stddev: 0.00004656758544388837",
            "extra": "mean: 1.0888618250064042 msec\nrounds: 40"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_dense_sum_n100",
            "value": 1685.4489634567337,
            "unit": "iter/sec",
            "range": "stddev: 0.000028378223340977004",
            "extra": "mean: 593.3137233352189 usec\nrounds: 600"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_dense_sum_n500",
            "value": 1343.56326821282,
            "unit": "iter/sec",
            "range": "stddev: 0.00002492484908838617",
            "extra": "mean: 744.2894753517483 usec\nrounds: 568"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_dense_matmul_n100",
            "value": 264.1128637934405,
            "unit": "iter/sec",
            "range": "stddev: 0.006117863256733758",
            "extra": "mean: 3.786260107277804 msec\nrounds: 261"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_mlp_layer",
            "value": 209.806881201913,
            "unit": "iter/sec",
            "range": "stddev: 0.006858042495005558",
            "extra": "mean: 4.766287903768154 msec\nrounds: 239"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_elementwise_chain",
            "value": 663.7426170516065,
            "unit": "iter/sec",
            "range": "stddev: 0.004358438402801063",
            "extra": "mean: 1.5066080952313616 msec\nrounds: 42"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_mixed_ops",
            "value": 1408.135776903817,
            "unit": "iter/sec",
            "range": "stddev: 0.00002761835749606136",
            "extra": "mean: 710.1587903680578 usec\nrounds: 353"
          }
        ]
      },
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
          "id": "aeaa6977713c1a6b47924e170adc64a1f692d395",
          "message": "Implement full ASD pipeline for reverse-mode Jacobian computation (#11)\n\n* Add sparse Jacobian computation with row-wise coloring\n\nImplement greedy row-wise matrix coloring and VJP-based sparse Jacobian\ncomputation. Rows that don't share non-zero columns can be evaluated\ntogether in a single VJP, reducing backward passes from m to the number\nof colors.\n\nNew public API:\n- color_rows(sparsity) -> (colors, num_colors)\n- sparse_jacobian(f, x, sparsity=None) -> csr_matrix\n\nCo-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>\n\n* Switch from scipy.sparse to jax.experimental.sparse.BCOO\n\n- jacobian_sparsity() returns BCOO instead of coo_matrix\n- sparse_jacobian() returns BCOO instead of csr_matrix\n- color_rows() accepts BCOO input\n- Update all tests (.toarray() → .todense(), .nnz → .nse)\n- Use int8 instead of bool for sparsity data (BCOO limitation)\n\nCo-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>\n\n* Update README\n\n* Rename main modules to detection.py, coloring.py, decompression.py\n\nRemove underscore prefix convention for the three main public modules:\n- _detect.py -> detection.py\n- _coloring.py -> coloring.py\n- _sparse_jacobian.py -> decompression.py\n\nThe decompression module now imports jacobian_sparsity from detection\ninstead of duplicating the detection logic.\n\nCo-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>\n\n* Rename test files and update CLAUDE.md documentation\n\nTest files renamed to match module names:\n- test_jacobian_sparsity.py -> test_detection.py\n- test_sparse_jacobian.py -> test_decompression.py\n\nUpdated CLAUDE.md files to reflect:\n- Full ASD pipeline (detection, coloring, decompression)\n- New module and test file structure\n- Added coloring and sparse_jacobian markers\n\nCo-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>\n\n* Update documentation with full ASD pipeline\n\nCo-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.5 <noreply@anthropic.com>",
          "timestamp": "2026-02-05T03:13:01+01:00",
          "tree_id": "ae32ffb00376513b13e0e25a0931265aa12b351e",
          "url": "https://github.com/adrhill/detex/commit/aeaa6977713c1a6b47924e170adc64a1f692d395"
        },
        "date": 1770257606919,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_bench_diagonal_n100",
            "value": 1055.0488903039582,
            "unit": "iter/sec",
            "range": "stddev: 0.00013116306364006273",
            "extra": "mean: 947.8233750019882 usec\nrounds: 8"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_diagonal_n500",
            "value": 343.4584896071303,
            "unit": "iter/sec",
            "range": "stddev: 0.006348204379840068",
            "extra": "mean: 2.9115599999984383 msec\nrounds: 21"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_diagonal_n1000",
            "value": 425.42487742315546,
            "unit": "iter/sec",
            "range": "stddev: 0.00017037879787891628",
            "extra": "mean: 2.3505912631558084 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_dense_sum_n100",
            "value": 1118.2780755972647,
            "unit": "iter/sec",
            "range": "stddev: 0.00010383944719810414",
            "extra": "mean: 894.2319641435399 usec\nrounds: 502"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_dense_sum_n500",
            "value": 590.5539499843768,
            "unit": "iter/sec",
            "range": "stddev: 0.0025046746978918347",
            "extra": "mean: 1.6933253939398003 msec\nrounds: 396"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_dense_matmul_n100",
            "value": 93.42865182763616,
            "unit": "iter/sec",
            "range": "stddev: 0.009201068198822637",
            "extra": "mean: 10.703354703703436 msec\nrounds: 27"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_mlp_layer",
            "value": 72.16680332931487,
            "unit": "iter/sec",
            "range": "stddev: 0.010382125853322354",
            "extra": "mean: 13.856786692307184 msec\nrounds: 26"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_elementwise_chain",
            "value": 843.6264904461575,
            "unit": "iter/sec",
            "range": "stddev: 0.00014053180756587788",
            "extra": "mean: 1.1853586999990284 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_mixed_ops",
            "value": 983.7836004790429,
            "unit": "iter/sec",
            "range": "stddev: 0.00007461371594399075",
            "extra": "mean: 1.0164837058811111 msec\nrounds: 34"
          }
        ]
      },
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
          "id": "6be7185765a25287933038a9ffa08f78b25021fa",
          "message": "Rename package from `detex` to `asdex` (#12)\n\nCo-authored-by: Claude Opus 4.5 <noreply@anthropic.com>",
          "timestamp": "2026-02-05T03:39:56+01:00",
          "tree_id": "0c18bbcc4054007f710c07b1539a4a264038d1e2",
          "url": "https://github.com/adrhill/asdex/commit/6be7185765a25287933038a9ffa08f78b25021fa"
        },
        "date": 1770259218847,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_bench_diagonal_n100",
            "value": 1001.4632809210294,
            "unit": "iter/sec",
            "range": "stddev: 0.00017279755942805586",
            "extra": "mean: 998.5388571414384 usec\nrounds: 7"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_diagonal_n500",
            "value": 264.3490066926449,
            "unit": "iter/sec",
            "range": "stddev: 0.009416161004423313",
            "extra": "mean: 3.7828778421047256 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_diagonal_n1000",
            "value": 415.16887372474133,
            "unit": "iter/sec",
            "range": "stddev: 0.00019584094282209228",
            "extra": "mean: 2.408658411764761 msec\nrounds: 17"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_dense_sum_n100",
            "value": 1167.220821450068,
            "unit": "iter/sec",
            "range": "stddev: 0.0000510730349776724",
            "extra": "mean: 856.7359163090276 usec\nrounds: 466"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_dense_sum_n500",
            "value": 556.3601799740507,
            "unit": "iter/sec",
            "range": "stddev: 0.0036480101163253427",
            "extra": "mean: 1.7973967871795593 msec\nrounds: 390"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_dense_matmul_n100",
            "value": 78.418128756086,
            "unit": "iter/sec",
            "range": "stddev: 0.013782088688828811",
            "extra": "mean: 12.752153307692774 msec\nrounds: 26"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_mlp_layer",
            "value": 58.858458236732005,
            "unit": "iter/sec",
            "range": "stddev: 0.01598009309053029",
            "extra": "mean: 16.989911559999484 msec\nrounds: 25"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_elementwise_chain",
            "value": 829.4922465970056,
            "unit": "iter/sec",
            "range": "stddev: 0.00018368318234093922",
            "extra": "mean: 1.2055567777788194 msec\nrounds: 18"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_mixed_ops",
            "value": 968.7228152160064,
            "unit": "iter/sec",
            "range": "stddev: 0.0001113231699407031",
            "extra": "mean: 1.0322870322580555 msec\nrounds: 31"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "committer": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "distinct": true,
          "id": "bc306f963e244ec0dbfcdf47a33c6734c4acf3a1",
          "message": "Fix equation in README",
          "timestamp": "2026-02-05T03:48:41+01:00",
          "tree_id": "d4e50fdaac57012353ca43aa837f199793f9bb76",
          "url": "https://github.com/adrhill/asdex/commit/bc306f963e244ec0dbfcdf47a33c6734c4acf3a1"
        },
        "date": 1770259748825,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_bench_diagonal_n100",
            "value": 1053.8646005290389,
            "unit": "iter/sec",
            "range": "stddev: 0.00012154662979510194",
            "extra": "mean: 948.8885000008552 usec\nrounds: 8"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_diagonal_n500",
            "value": 331.19074475163,
            "unit": "iter/sec",
            "range": "stddev: 0.006773630114375006",
            "extra": "mean: 3.0194080476189944 msec\nrounds: 21"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_diagonal_n1000",
            "value": 426.9296433427451,
            "unit": "iter/sec",
            "range": "stddev: 0.00010494920735620508",
            "extra": "mean: 2.34230631579074 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_dense_sum_n100",
            "value": 1160.6004177094346,
            "unit": "iter/sec",
            "range": "stddev: 0.000026721177098177307",
            "extra": "mean: 861.6229881888237 usec\nrounds: 508"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_dense_sum_n500",
            "value": 570.029664343726,
            "unit": "iter/sec",
            "range": "stddev: 0.0026170501602643653",
            "extra": "mean: 1.7542946666666865 msec\nrounds: 393"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_dense_matmul_n100",
            "value": 95.03779038833748,
            "unit": "iter/sec",
            "range": "stddev: 0.0085028268618674",
            "extra": "mean: 10.522130153845776 msec\nrounds: 26"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_mlp_layer",
            "value": 72.28431715760392,
            "unit": "iter/sec",
            "range": "stddev: 0.010596249789293327",
            "extra": "mean: 13.834259481481528 msec\nrounds: 27"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_elementwise_chain",
            "value": 841.1799736195716,
            "unit": "iter/sec",
            "range": "stddev: 0.0000882652520661227",
            "extra": "mean: 1.1888062380955537 msec\nrounds: 21"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_mixed_ops",
            "value": 987.102518493554,
            "unit": "iter/sec",
            "range": "stddev: 0.0000642172524788849",
            "extra": "mean: 1.0130659999998068 msec\nrounds: 33"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "committer": {
            "email": "adrian.hill@mailbox.org",
            "name": "adrhill",
            "username": "adrhill"
          },
          "distinct": true,
          "id": "64f352c4f8eb3de69855497b351852d1705e3073",
          "message": "Fix equation in README",
          "timestamp": "2026-02-05T03:50:15+01:00",
          "tree_id": "66ae79f30224bf954ac3b6d85cc20bbc92957d02",
          "url": "https://github.com/adrhill/asdex/commit/64f352c4f8eb3de69855497b351852d1705e3073"
        },
        "date": 1770259842176,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_bench_diagonal_n100",
            "value": 1054.0591356189846,
            "unit": "iter/sec",
            "range": "stddev: 0.00011045777503323493",
            "extra": "mean: 948.7133749974674 usec\nrounds: 8"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_diagonal_n500",
            "value": 359.3601520621164,
            "unit": "iter/sec",
            "range": "stddev: 0.00561736868912918",
            "extra": "mean: 2.7827236666661563 msec\nrounds: 21"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_diagonal_n1000",
            "value": 385.31655935439824,
            "unit": "iter/sec",
            "range": "stddev: 0.0005603779828119075",
            "extra": "mean: 2.595268684209965 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_dense_sum_n100",
            "value": 1132.9866756429315,
            "unit": "iter/sec",
            "range": "stddev: 0.00009359651318311288",
            "extra": "mean: 882.6229129592668 usec\nrounds: 517"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_dense_sum_n500",
            "value": 577.2509056089783,
            "unit": "iter/sec",
            "range": "stddev: 0.0028172732800779954",
            "extra": "mean: 1.7323489496218927 msec\nrounds: 397"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_dense_matmul_n100",
            "value": 98.11741141544431,
            "unit": "iter/sec",
            "range": "stddev: 0.008229094719794728",
            "extra": "mean: 10.191870999998615 msec\nrounds: 29"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_mlp_layer",
            "value": 71.94954341606163,
            "unit": "iter/sec",
            "range": "stddev: 0.010766116802458953",
            "extra": "mean: 13.898628851851273 msec\nrounds: 27"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_elementwise_chain",
            "value": 848.6247258425445,
            "unit": "iter/sec",
            "range": "stddev: 0.00006874798446348126",
            "extra": "mean: 1.1783771666647642 msec\nrounds: 18"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_mixed_ops",
            "value": 987.369874876932,
            "unit": "iter/sec",
            "range": "stddev: 0.00006050153864928073",
            "extra": "mean: 1.012791685714173 msec\nrounds: 35"
          }
        ]
      },
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
          "id": "b9ffbc9eab4d98bf4f10276c8af42e0929e4fa7d",
          "message": "Fix handling of `constvars` in Jaxpr interpreter (#13)\n\n* Fix empty array handling in concatenate sparsity detection\n\nInitialize jaxpr.constvars in prop_jaxpr so that constant arrays\n(like empty arrays) have the correct number of dependency sets.\nPreviously, uninitialized constvars fell back to [set()] regardless\nof their actual size, causing shifted indices in concatenate.\n\nCo-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>\n\n* Add tests for constvar initialization in propagation handlers\n\nTest constant variable handling across all indexing handlers (slice,\nsqueeze, broadcast, concatenate, reshape) to prevent regressions after\nthe empty array fix. Also clean up TODO.md by removing resolved items.\n\nCo-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.5 <noreply@anthropic.com>",
          "timestamp": "2026-02-05T13:37:01+01:00",
          "tree_id": "8efc943d61ecf4a04d6d0b87b710c1e49ad10e32",
          "url": "https://github.com/adrhill/asdex/commit/b9ffbc9eab4d98bf4f10276c8af42e0929e4fa7d"
        },
        "date": 1770295048756,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_bench_diagonal_n100",
            "value": 1069.064507086966,
            "unit": "iter/sec",
            "range": "stddev: 0.00011709795382738916",
            "extra": "mean: 935.3972499983598 usec\nrounds: 8"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_diagonal_n500",
            "value": 357.70066334395824,
            "unit": "iter/sec",
            "range": "stddev: 0.005825574673441266",
            "extra": "mean: 2.795633619047608 msec\nrounds: 21"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_diagonal_n1000",
            "value": 418.7497625628694,
            "unit": "iter/sec",
            "range": "stddev: 0.0002396801455593688",
            "extra": "mean: 2.388061055556692 msec\nrounds: 18"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_dense_sum_n100",
            "value": 1164.7199043916123,
            "unit": "iter/sec",
            "range": "stddev: 0.00002214172002309312",
            "extra": "mean: 858.5755220885891 usec\nrounds: 498"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_dense_sum_n500",
            "value": 581.646365788406,
            "unit": "iter/sec",
            "range": "stddev: 0.0026007580635606965",
            "extra": "mean: 1.7192577119338255 msec\nrounds: 243"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_dense_matmul_n100",
            "value": 94.3242240848029,
            "unit": "iter/sec",
            "range": "stddev: 0.009159272246236699",
            "extra": "mean: 10.601730464286062 msec\nrounds: 28"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_mlp_layer",
            "value": 69.96428628954625,
            "unit": "iter/sec",
            "range": "stddev: 0.011717742198034582",
            "extra": "mean: 14.2930065185188 msec\nrounds: 27"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_elementwise_chain",
            "value": 844.0236311419565,
            "unit": "iter/sec",
            "range": "stddev: 0.0001307483597271736",
            "extra": "mean: 1.184800950000664 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_mixed_ops",
            "value": 992.0572887216232,
            "unit": "iter/sec",
            "range": "stddev: 0.00008009884375040523",
            "extra": "mean: 1.0080063030317654 msec\nrounds: 33"
          }
        ]
      },
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
          "id": "9ecdb54a2d571f9d523f4578081e203655bec72b",
          "message": "Add Hessian sparsity detection and computation (#14)\n\n* Rename `_propagate` to `_interpret`\n\nAligns with JAX's custom interpreter terminology.\n\nCo-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>\n\n* Add Hessian sparsity detection via interpreter composition\n\nImplement `hessian_sparsity(f, n)` which computes second-order sparsity\nby analyzing the Jacobian sparsity of the gradient function. This\ndemonstrates how our sparsity interpreter composes with JAX's autodiff.\n\nChanges:\n- Add `add_any` primitive handler (used in gradient jaxprs)\n- Add `hessian_sparsity` function to detection.py\n- Export in public API\n- Add tests in test_detection.py and test_sympy.py\n\nCo-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>\n\n* Update README with Hessian sparsity documentation\n\n- Add mention of Hessians in intro section\n- Add ### Jacobians and ### Hessians subsections to examples\n- Add Hessian sparsity paragraph to \"How it works\" section\n\nCo-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>\n\n* Add sparse_hessian function for computing sparse Hessians\n\nComputes the Hessian by applying sparse_jacobian to the gradient function,\ndemonstrating how sparse differentiation composes with JAX's autodiff.\n\nCo-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>\n\n* Use HVP for sparse Hessian computation\n\nReplace VJP-on-gradient with forward-over-reverse HVP for ~30% speedup.\nUpdate README with sparse_hessian examples and HVP explanation.\n\nCo-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>\n\n* Add pytest markers for test categories\n\nRegister markers in pyproject.toml and add @pytest.mark.hessian\nto Hessian-related tests.\n\nCo-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.5 <noreply@anthropic.com>",
          "timestamp": "2026-02-05T19:10:07+01:00",
          "tree_id": "37e155464d7b87b064fd80056226e18b8a029c90",
          "url": "https://github.com/adrhill/asdex/commit/9ecdb54a2d571f9d523f4578081e203655bec72b"
        },
        "date": 1770315040062,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_bench_diagonal_n100",
            "value": 1054.0774681141997,
            "unit": "iter/sec",
            "range": "stddev: 0.00010920189620295607",
            "extra": "mean: 948.6968749925495 usec\nrounds: 8"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_diagonal_n500",
            "value": 325.68142120258574,
            "unit": "iter/sec",
            "range": "stddev: 0.0067244696350223475",
            "extra": "mean: 3.0704852499951585 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_diagonal_n1000",
            "value": 424.7325520588975,
            "unit": "iter/sec",
            "range": "stddev: 0.00014544143252997788",
            "extra": "mean: 2.354422789476542 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_dense_sum_n100",
            "value": 1163.5890008009867,
            "unit": "iter/sec",
            "range": "stddev: 0.00002724563278913846",
            "extra": "mean: 859.4099800802724 usec\nrounds: 502"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_dense_sum_n500",
            "value": 577.8482983226987,
            "unit": "iter/sec",
            "range": "stddev: 0.002784161878609437",
            "extra": "mean: 1.7305580078762317 msec\nrounds: 381"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_dense_matmul_n100",
            "value": 93.48310282011347,
            "unit": "iter/sec",
            "range": "stddev: 0.009320643287222194",
            "extra": "mean: 10.697120333331979 msec\nrounds: 27"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_mlp_layer",
            "value": 68.55961612625104,
            "unit": "iter/sec",
            "range": "stddev: 0.01216060128804101",
            "extra": "mean: 14.58584596154275 msec\nrounds: 26"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_elementwise_chain",
            "value": 836.7809755469465,
            "unit": "iter/sec",
            "range": "stddev: 0.000117638544680392",
            "extra": "mean: 1.1950558500046782 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_bench_mixed_ops",
            "value": 981.1006129398843,
            "unit": "iter/sec",
            "range": "stddev: 0.00008231375930930491",
            "extra": "mean: 1.019263454543651 msec\nrounds: 33"
          }
        ]
      },
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
          "id": "c71c658ab58a0565403ced62a3e1f434d3a2039e",
          "message": "Redesign benchmarks with realistic test functions (#15)\n\n* Redesign benchmarks with realistic test functions\n\nReplace synthetic benchmarks with 3 realistic test cases:\n- Heat equation RHS: tridiagonal Jacobian (~3 colors, 98% sparse)\n- Pure ConvNet: 3 conv layers with ReLU (~18 colors, 95% sparse)\n- Rosenbrock function: sparse Hessian (~3 colors, 98% sparse)\n\nEach function now has 4 benchmark phases: detection, coloring,\nmaterialization (with pre-computed sparsity/colors), and end-to-end.\n\nAlso add TODO for custom_vjp/custom_jvp support (needed for jax.nn.relu).\n\nCo-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>\n\n* Add Hessian tests and improve coverage to 99%\n\nAdd 6 new tests for sparse_hessian:\n- test_hessian_quadratic: basic quadratic function\n- test_hessian_rosenbrock: sparse tridiagonal pattern\n- test_hessian_precomputed_sparsity: pre-computed sparsity\n- test_hessian_precomputed_colors: pre-computed colors\n- test_hessian_zero: linear function (zero Hessian)\n- test_hessian_single_input: single dimension\n\nAlso move hessian_sparsity import to module level in decompression.py.\n\nCo-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.5 <noreply@anthropic.com>",
          "timestamp": "2026-02-06T15:18:46+01:00",
          "tree_id": "d4fc63846af52bfae27dce0aa469a6234eaff27f",
          "url": "https://github.com/adrhill/asdex/commit/c71c658ab58a0565403ced62a3e1f434d3a2039e"
        },
        "date": 1770387596647,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 444.8577397523808,
            "unit": "iter/sec",
            "range": "stddev: 0.00019287888722053595",
            "extra": "mean: 2.247909636362909 msec\nrounds: 22"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 1279.1086699046168,
            "unit": "iter/sec",
            "range": "stddev: 0.00001873059124719543",
            "extra": "mean: 781.7944038129067 usec\nrounds: 1154"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 25.681951608896146,
            "unit": "iter/sec",
            "range": "stddev: 0.006590553478836575",
            "extra": "mean: 38.937850799999296 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 29.61872394234056,
            "unit": "iter/sec",
            "range": "stddev: 0.0004703285107900902",
            "extra": "mean: 33.7624268333343 msec\nrounds: 12"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 19.789696478696747,
            "unit": "iter/sec",
            "range": "stddev: 0.014393870357419097",
            "extra": "mean: 50.531345999999644 msec\nrounds: 12"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 144.84349241414247,
            "unit": "iter/sec",
            "range": "stddev: 0.000059335944312373055",
            "extra": "mean: 6.904003647887466 msec\nrounds: 142"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 2.74565583610344,
            "unit": "iter/sec",
            "range": "stddev: 0.0015947026986007455",
            "extra": "mean: 364.21170739999695 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 2.3563527405611375,
            "unit": "iter/sec",
            "range": "stddev: 0.018450783950387448",
            "extra": "mean: 424.3846784000013 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 10.543084128788397,
            "unit": "iter/sec",
            "range": "stddev: 0.03464758780697518",
            "extra": "mean: 94.84890642857076 msec\nrounds: 7"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 4.681852591391767,
            "unit": "iter/sec",
            "range": "stddev: 0.003212477475886659",
            "extra": "mean: 213.5906631999987 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 0.43612065135260547,
            "unit": "iter/sec",
            "range": "stddev: 0.012616653719753667",
            "extra": "mean: 2.2929434708 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 0.3839516882670346,
            "unit": "iter/sec",
            "range": "stddev: 0.03348593097557271",
            "extra": "mean: 2.604494342799998 sec\nrounds: 5"
          }
        ]
      },
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
          "id": "ced92bb4520950b819640b6457dbbee4b71cae07",
          "message": "Add `SparsityPattern` data structure (#16)\n\n* Add SparsityPattern data structure to replace BCOO in pipeline\n\nReplace BCOO with a custom SparsityPattern class optimized for the\ndetection->coloring->decompression pipeline:\n\n- Store row/col indices separately for direct access (no slicing)\n- Cache col_to_rows mapping for coloring algorithm\n- No unnecessary data values (pattern-only, no all-1s array)\n- Julia-style visualization: dots for small matrices, braille for large\n\nCo-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>\n\n* Update README to showcase SparsityPattern visualization\n\n- Jacobian example: n=100 with braille display\n- Hessian example: n=5 diagonal pattern with dot display\n- Show SparsityPattern pretty-printing output\n\nCo-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>\n\n* Use Julia-style bracket characters for braille frame\n\nBox-drawing characters (┌─┐│└┘) have inconsistent width with braille\nin some fonts. Switch to mathematical bracket characters (⎡⎢⎣⎤⎥⎦)\nwhich align correctly, matching Julia's SparseArrays visualization.\n\nCo-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.5 <noreply@anthropic.com>",
          "timestamp": "2026-02-06T17:07:14+01:00",
          "tree_id": "73e1e1c2de7a7d140f94137e699577d78d38177e",
          "url": "https://github.com/adrhill/asdex/commit/ced92bb4520950b819640b6457dbbee4b71cae07"
        },
        "date": 1770394106346,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 658.6643183407812,
            "unit": "iter/sec",
            "range": "stddev: 0.0038036857844234674",
            "extra": "mean: 1.5182240363635697 msec\nrounds: 165"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3386.9774510275793,
            "unit": "iter/sec",
            "range": "stddev: 0.0000273161587884041",
            "extra": "mean: 295.2484964718643 usec\nrounds: 1984"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 33.45235659508205,
            "unit": "iter/sec",
            "range": "stddev: 0.000571204862102254",
            "extra": "mean: 29.893260200000782 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 31.76640590225183,
            "unit": "iter/sec",
            "range": "stddev: 0.00020689727169111754",
            "extra": "mean: 31.479796709677903 msec\nrounds: 31"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 21.015167176784573,
            "unit": "iter/sec",
            "range": "stddev: 0.013426086562642568",
            "extra": "mean: 47.58467974999974 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 245.24665497798958,
            "unit": "iter/sec",
            "range": "stddev: 0.00021621249484248735",
            "extra": "mean: 4.077527581730924 msec\nrounds: 208"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 2.7824130863022223,
            "unit": "iter/sec",
            "range": "stddev: 0.002020764051644127",
            "extra": "mean: 359.4002647999986 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 2.446697402777567,
            "unit": "iter/sec",
            "range": "stddev: 0.0026561263904926846",
            "extra": "mean: 408.71421159999954 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 44.85053583281214,
            "unit": "iter/sec",
            "range": "stddev: 0.009383748701236745",
            "extra": "mean: 22.296277657142536 msec\nrounds: 35"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 5.566664731879604,
            "unit": "iter/sec",
            "range": "stddev: 0.0007132352750473701",
            "extra": "mean: 179.6407809999986 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 0.4377555001289671,
            "unit": "iter/sec",
            "range": "stddev: 0.022375799519721178",
            "extra": "mean: 2.284380207000004 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 0.40402917248649256,
            "unit": "iter/sec",
            "range": "stddev: 0.03129882212142588",
            "extra": "mean: 2.475068802199999 sec\nrounds: 5"
          }
        ]
      }
    ]
  }
}