window.BENCHMARK_DATA = {
  "lastUpdate": 1770257607947,
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
      }
    ]
  }
}