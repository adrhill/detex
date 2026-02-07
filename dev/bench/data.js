window.BENCHMARK_DATA = {
  "lastUpdate": 1770493736053,
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
          "id": "e6443d223df8fe6343fe597a27ebeb1a642186df",
          "message": "Support gather, scatter, and custom JVP/VJP (#17)\n\n- Add prop_gather and prop_scatter with static index support\n- Add prop_custom_call to trace into custom_jvp_call/custom_vjp_call\n- Add const_vals tracking to propagate constant values through primitives\n- This enables precise sparsity patterns for static indexing operations\n  (e.g., x[[2,0,1]] produces permutation matrix, not all-ones)\n\n---------\n\nCo-authored-by: Claude Opus 4.5 <noreply@anthropic.com>",
          "timestamp": "2026-02-07T01:46:33+01:00",
          "tree_id": "8983067a190da81e21583bf6658b1b368c74b0f1",
          "url": "https://github.com/adrhill/asdex/commit/e6443d223df8fe6343fe597a27ebeb1a642186df"
        },
        "date": 1770425266154,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 694.1697879670656,
            "unit": "iter/sec",
            "range": "stddev: 0.0034795646168398415",
            "extra": "mean: 1.4405697530118444 msec\nrounds: 166"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3380.2252265282987,
            "unit": "iter/sec",
            "range": "stddev: 0.00000717572099927998",
            "extra": "mean: 295.83827496224626 usec\nrounds: 1993"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 32.66335202019078,
            "unit": "iter/sec",
            "range": "stddev: 0.0006772045378707155",
            "extra": "mean: 30.615351399998758 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 30.091325574516297,
            "unit": "iter/sec",
            "range": "stddev: 0.00938402662080791",
            "extra": "mean: 33.23216843750076 msec\nrounds: 32"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 22.997422878895645,
            "unit": "iter/sec",
            "range": "stddev: 0.011445354025197571",
            "extra": "mean: 43.483133099999804 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 238.67453114080476,
            "unit": "iter/sec",
            "range": "stddev: 0.00002841045992376862",
            "extra": "mean: 4.18980607281493 msec\nrounds: 206"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 2.4775076237232847,
            "unit": "iter/sec",
            "range": "stddev: 0.0029116580244819743",
            "extra": "mean: 403.6314521999998 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 2.2149689441441436,
            "unit": "iter/sec",
            "range": "stddev: 0.004243021060255774",
            "extra": "mean: 451.4735986000005 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 43.53190475705649,
            "unit": "iter/sec",
            "range": "stddev: 0.013760512826358403",
            "extra": "mean: 22.97165735294183 msec\nrounds: 34"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 5.30469303159988,
            "unit": "iter/sec",
            "range": "stddev: 0.001979586878893617",
            "extra": "mean: 188.51232183333386 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 0.44794935019352644,
            "unit": "iter/sec",
            "range": "stddev: 0.012377084831477118",
            "extra": "mean: 2.2323952464000056 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 0.40845881396684586,
            "unit": "iter/sec",
            "range": "stddev: 0.025552567374849934",
            "extra": "mean: 2.448227252800007 sec\nrounds: 5"
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
          "id": "74543ac283f7bb7097cb432c38a7eb5c8eef61a1",
          "message": "Add 28 trivial primitives (#18)\n\n* Add ~28 trivial primitives to prop_dispatch\n\nExtend interpreter coverage with primitives that map directly to\nexisting handler patterns: 15 unary elementwise (inverse trig, cbrt,\nrsqrt, square, exp2, logistic, etc.), 4 binary elementwise (atan2, rem,\nnextafter, complex), 3 bitwise ops, 4 zero-derivative (argmax, argmin,\nclz, population_count), 3 identity pass-through (bitcast_convert_type,\nreduce_precision, stop_gradient), and a dedicated iota handler with\nconst-value tracking for downstream gather/scatter precision.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Increase test coverage to 99%\n\nAdd tests for edge cases and defensive branches:\n- Error paths for missing jaxpr/call_jaxpr params and unknown primitives\n- prop_jaxpr default const_vals, stop_gradient, reshape size mismatch\n- Dynamic and 2D scatter conservative fallback\n- SparsityPattern validation, empty to_bcoo, large zero-dim braille\n- Extend SymPy random expressions with log, atan, asinh\n\nThe only uncovered lines (5/680) are unreachable scatter branches\nwhere JAX validates index/updates shape consistency before our code.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-07T02:38:13+01:00",
          "tree_id": "5de5c6f92f46fb77be9a2219511ca4ee367370bf",
          "url": "https://github.com/adrhill/asdex/commit/74543ac283f7bb7097cb432c38a7eb5c8eef61a1"
        },
        "date": 1770428360610,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 736.2016470094069,
            "unit": "iter/sec",
            "range": "stddev: 0.0032230216490377315",
            "extra": "mean: 1.35832350289108 msec\nrounds: 173"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3598.5782161012066,
            "unit": "iter/sec",
            "range": "stddev: 0.000014295369922670833",
            "extra": "mean: 277.88752666974847 usec\nrounds: 2081"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 35.55063673785402,
            "unit": "iter/sec",
            "range": "stddev: 0.0005334772689661583",
            "extra": "mean: 28.12889140000152 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 34.06552626613208,
            "unit": "iter/sec",
            "range": "stddev: 0.00020895368934754175",
            "extra": "mean: 29.35519011764686 msec\nrounds: 34"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 23.019196270095,
            "unit": "iter/sec",
            "range": "stddev: 0.013898650916275143",
            "extra": "mean: 43.44200328571562 msec\nrounds: 21"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 267.47898505977855,
            "unit": "iter/sec",
            "range": "stddev: 0.00009020860668547253",
            "extra": "mean: 3.7386114642857318 msec\nrounds: 224"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 2.549564101051154,
            "unit": "iter/sec",
            "range": "stddev: 0.0012235878756469702",
            "extra": "mean: 392.2239097999977 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 2.2788382011040076,
            "unit": "iter/sec",
            "range": "stddev: 0.0026344636648777645",
            "extra": "mean: 438.82009680000067 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 49.745322444144506,
            "unit": "iter/sec",
            "range": "stddev: 0.010637518482229845",
            "extra": "mean: 20.10239256410146 msec\nrounds: 39"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 5.682975248494706,
            "unit": "iter/sec",
            "range": "stddev: 0.0001725820791701896",
            "extra": "mean: 175.96416599999762 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 0.4969696530448073,
            "unit": "iter/sec",
            "range": "stddev: 0.014096172558825823",
            "extra": "mean: 2.012195299800004 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 0.44706400057682355,
            "unit": "iter/sec",
            "range": "stddev: 0.01239900780296027",
            "extra": "mean: 2.2368162023999956 sec\nrounds: 5"
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
          "id": "8e7a89254eca1fef6ff552de025165ae85d3a22f",
          "message": "Fix braille rendering and update README (#19)\n\nRewrite _render_braille to use Julia-style linear interpolation instead\nof block-OR downsampling, which made sparse patterns look artificially\ndense. Each non-zero is now scaled individually to the output grid,\nmatching Julia's SparseArrays braille display.\n\nAlso update README: use n=50 example, fix typos, add\nSparseMatrixColorings.jl to related work.\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-07T03:12:43+01:00",
          "tree_id": "da9db562770e5d87075ef0236b6b7e7b686921c9",
          "url": "https://github.com/adrhill/asdex/commit/8e7a89254eca1fef6ff552de025165ae85d3a22f"
        },
        "date": 1770430434170,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 632.6359101279828,
            "unit": "iter/sec",
            "range": "stddev: 0.004989920799901693",
            "extra": "mean: 1.5806880134226005 msec\nrounds: 149"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3375.452495076986,
            "unit": "iter/sec",
            "range": "stddev: 0.000019369329820816824",
            "extra": "mean: 296.2565764022677 usec\nrounds: 2068"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 33.47028951847288,
            "unit": "iter/sec",
            "range": "stddev: 0.0006378333460978401",
            "extra": "mean: 29.87724380000003 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 32.05348280668388,
            "unit": "iter/sec",
            "range": "stddev: 0.0003232185976418458",
            "extra": "mean: 31.19785784374973 msec\nrounds: 32"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 22.4763094539022,
            "unit": "iter/sec",
            "range": "stddev: 0.012531810323954727",
            "extra": "mean: 44.49128990909075 msec\nrounds: 11"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 245.06784583679834,
            "unit": "iter/sec",
            "range": "stddev: 0.00004461178809096297",
            "extra": "mean: 4.080502672986095 msec\nrounds: 211"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 2.413456895420666,
            "unit": "iter/sec",
            "range": "stddev: 0.009399210465201339",
            "extra": "mean: 414.3434266000014 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 2.0929974236503535,
            "unit": "iter/sec",
            "range": "stddev: 0.02821716685887922",
            "extra": "mean: 477.78367460000055 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 42.51216505341042,
            "unit": "iter/sec",
            "range": "stddev: 0.014668885374465038",
            "extra": "mean: 23.522678714284343 msec\nrounds: 35"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 5.307858176014629,
            "unit": "iter/sec",
            "range": "stddev: 0.005045169364548127",
            "extra": "mean: 188.39990950000166 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 0.4430832314347581,
            "unit": "iter/sec",
            "range": "stddev: 0.021892536267014966",
            "extra": "mean: 2.256912311399998 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 0.40166510174743103,
            "unit": "iter/sec",
            "range": "stddev: 0.0462404923428099",
            "extra": "mean: 2.4896362557999994 sec\nrounds: 5"
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
          "id": "d8abf294b5d4ddae8c812d2e644ef0ac8d8c476a",
          "message": "Support multi-dimensional input and output shapes (#20)\n\nRename `n` parameter to `input_shape: int | tuple[int, ...]` in\n`jacobian_sparsity` and `hessian_sparsity`, allowing functions on\nmatrices and higher-dimensional arrays. Update `sparse_jacobian` and\n`sparse_hessian` to reshape VJP seeds and HVP tangents to match the\nactual input/output shapes, flattening the results for the sparse matrix.\n\nAdd `tests/test_multidim.py` with 15 tests including a tiny LeNet convnet\non 2D image input and multi-dimensional output correctness checks.\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-07T03:59:33+01:00",
          "tree_id": "fcd47f0333d6b530b2b096b90a97b33d8c9cce83",
          "url": "https://github.com/adrhill/asdex/commit/d8abf294b5d4ddae8c812d2e644ef0ac8d8c476a"
        },
        "date": 1770433247037,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 677.0611109918782,
            "unit": "iter/sec",
            "range": "stddev: 0.004061364787030313",
            "extra": "mean: 1.476971552146931 msec\nrounds: 163"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3386.6689157342335,
            "unit": "iter/sec",
            "range": "stddev: 0.0000070882499742896586",
            "extra": "mean: 295.27539446034064 usec\nrounds: 2094"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 32.40890601922615,
            "unit": "iter/sec",
            "range": "stddev: 0.0005419426385338814",
            "extra": "mean: 30.855716000002076 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 30.88782834943404,
            "unit": "iter/sec",
            "range": "stddev: 0.00032284454246964475",
            "extra": "mean: 32.37521229032352 msec\nrounds: 31"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 20.801689971334365,
            "unit": "iter/sec",
            "range": "stddev: 0.01978838608618007",
            "extra": "mean: 48.073017210526814 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 242.23554974659962,
            "unit": "iter/sec",
            "range": "stddev: 0.00020534673833592633",
            "extra": "mean: 4.128213224880043 msec\nrounds: 209"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 2.393174173306563,
            "unit": "iter/sec",
            "range": "stddev: 0.007365247221694762",
            "extra": "mean: 417.855086000003 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 2.1573432906933694,
            "unit": "iter/sec",
            "range": "stddev: 0.002967856038761839",
            "extra": "mean: 463.53308920000416 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 42.20352927158153,
            "unit": "iter/sec",
            "range": "stddev: 0.015042547078038955",
            "extra": "mean: 23.69470082857187 msec\nrounds: 35"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 5.072152789405264,
            "unit": "iter/sec",
            "range": "stddev: 0.0010596527878468382",
            "extra": "mean: 197.1549441666672 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 0.4389887497409245,
            "unit": "iter/sec",
            "range": "stddev: 0.010187656164640688",
            "extra": "mean: 2.2779627054000002 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 0.4007114364841244,
            "unit": "iter/sec",
            "range": "stddev: 0.008588270173646582",
            "extra": "mean: 2.4955614163999997 sec\nrounds: 5"
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
          "id": "b9c2deaedc549dbf6297e04dac3dc07ea9c224e0",
          "message": "Improve coloring and API (#21)\n\n* Add LargestFirst ordering, column coloring, and star coloring\n\nRefactor greedy coloring into _greedy_color helper with LargestFirst\nvertex ordering (sort by decreasing degree) for fewer colors. Add\ncolor_cols for column coloring + JVP-based Jacobian computation, with\nauto direction selection in sparse_jacobian. Add star_color for\nsymmetric Hessian coloring (Gebremedhin et al. 2005) with symmetric\ndecompression, used by default in sparse_hessian.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Add SparseMatrixColorings.jl attribution\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Add SMC-sourced test cases for coloring algorithms\n\nPort test matrices from SparseMatrixColorings.jl: Gebremedhin et al.\nFigures 4.1 and 6.1, banded matrices with known star chromatic numbers,\nanti-diagonal, triangle, bidiagonal, and small hand-crafted patterns.\nTighten arrow matrix assertions to exact counts verified against SMC.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Add unified color() function and ColoringResult dataclass\n\nReplace the separate colors/partition parameters on sparse_jacobian\nwith a single coloring: ColoringResult that carries the color array,\ncount, and partition together. The new color(sparsity) function\nauto-picks the best of row/column coloring (ties favor column/JVPs).\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Rename ColoringResult to ColoredPattern and simplify sparse_jacobian API\n\nBundle the sparsity pattern into ColoredPattern so callers pass a single\nobject instead of separate sparsity and coloring arguments.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Add coloring convenience functions, compressed visualization, and rename API\n\n- Add jacobian_coloring() and hessian_coloring() one-step convenience functions\n- Add ColoredPattern._compressed_pattern() and side-by-side/stacked __str__\n- Extract SparsityPattern._render() helper for reuse by ColoredPattern\n- Move ColoredPattern from coloring.py to pattern.py\n- Rename sparse_jacobian → jacobian, sparse_hessian → hessian\n- Add colored_pattern parameter to hessian() matching jacobian() API\n- Update README, CLAUDE.md, exports, tests, and pytest markers\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Update README to describe automatic coloring and AD mode selection\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Rename color() to color_jacobian_pattern() and add color_hessian_pattern()\n\nThe generic name `color()` was misleading since it only handled Jacobians.\nThe new names clarify intent and `color_hessian_pattern` wraps star_color\nwith the same nnz==0 early-return guard, simplifying callers.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Fix SIM108 lint warning and add test for size-0 binary elementwise\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Fix printing\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-07T16:52:31+01:00",
          "tree_id": "8f40488aafe282568526eaffc4c53420aa98cb6c",
          "url": "https://github.com/adrhill/asdex/commit/b9c2deaedc549dbf6297e04dac3dc07ea9c224e0"
        },
        "date": 1770479625958,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 734.8841250077743,
            "unit": "iter/sec",
            "range": "stddev: 0.002764833940172751",
            "extra": "mean: 1.3607587454544907 msec\nrounds: 165"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3254.325339306064,
            "unit": "iter/sec",
            "range": "stddev: 0.000007946382942931392",
            "extra": "mean: 307.28335238087624 usec\nrounds: 1995"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 32.84250602193617,
            "unit": "iter/sec",
            "range": "stddev: 0.00043448072692210796",
            "extra": "mean: 30.448346400000048 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 49.2691786594981,
            "unit": "iter/sec",
            "range": "stddev: 0.014330025408853732",
            "extra": "mean: 20.296664714284216 msec\nrounds: 7"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 21.284304365637617,
            "unit": "iter/sec",
            "range": "stddev: 0.01157854476877638",
            "extra": "mean: 46.982977823529296 msec\nrounds: 17"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 242.42354962408476,
            "unit": "iter/sec",
            "range": "stddev: 0.0000316790361878196",
            "extra": "mean: 4.125011788461372 msec\nrounds: 208"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 2.4969751006901695,
            "unit": "iter/sec",
            "range": "stddev: 0.0005464811702785462",
            "extra": "mean: 400.4845701999983 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 6.122329051911907,
            "unit": "iter/sec",
            "range": "stddev: 0.001691570595730755",
            "extra": "mean: 163.33653279999965 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 47.686912908530985,
            "unit": "iter/sec",
            "range": "stddev: 0.010106792125336104",
            "extra": "mean: 20.970114000000706 msec\nrounds: 37"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 5.5666622322017,
            "unit": "iter/sec",
            "range": "stddev: 0.000429932547297812",
            "extra": "mean: 179.64086166666604 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 0.43661173848502344,
            "unit": "iter/sec",
            "range": "stddev: 0.00391796045611275",
            "extra": "mean: 2.290364440200001 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 0.275065790062376,
            "unit": "iter/sec",
            "range": "stddev: 0.015609194218410213",
            "extra": "mean: 3.635493893199995 sec\nrounds: 5"
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
          "id": "c77d21d1b704401c27825740687ee31cdff8831d",
          "message": "Pre-allocate decompression buffers (#22)\n\n* Pre-allocate decompression buffers and vectorize extraction\n\nCache BCOO indices, extraction index arrays, and seed matrices on\nColoredPattern so that repeated calls to jacobian/hessian with the\nsame colored pattern skip all redundant Python-side work.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Add decompression benchmarks to dashboard\n\nAdd a `dashboard` marker for the 15 benchmarks tracked in the GitHub\nPages dashboard. Each group now covers detection, coloring,\ndecompression, materialization, and end-to-end.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-07T20:47:39+01:00",
          "tree_id": "ee398ac68726784184f03a22b5a067241e7fff23",
          "url": "https://github.com/adrhill/asdex/commit/c77d21d1b704401c27825740687ee31cdff8831d"
        },
        "date": 1770493735765,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 735.1467054168455,
            "unit": "iter/sec",
            "range": "stddev: 0.002595760710259541",
            "extra": "mean: 1.360272708333742 msec\nrounds: 168"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3267.362485235797,
            "unit": "iter/sec",
            "range": "stddev: 0.000014058811323941864",
            "extra": "mean: 306.0572570440811 usec\nrounds: 2023"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_decompression",
            "value": 6971.952671046768,
            "unit": "iter/sec",
            "range": "stddev: 0.00004350945261481345",
            "extra": "mean: 143.43183999983466 usec\nrounds: 25"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 33.48923308095324,
            "unit": "iter/sec",
            "range": "stddev: 0.00015061770290838248",
            "extra": "mean: 29.860343400002876 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 67.04596805661262,
            "unit": "iter/sec",
            "range": "stddev: 0.0005076461986959546",
            "extra": "mean: 14.915140000001413 msec\nrounds: 7"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 21.473529459304483,
            "unit": "iter/sec",
            "range": "stddev: 0.014990568318235932",
            "extra": "mean: 46.56896305263408 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 243.11194672709567,
            "unit": "iter/sec",
            "range": "stddev: 0.000052866711172762246",
            "extra": "mean: 4.113331382774644 msec\nrounds: 209"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_decompression",
            "value": 6480.901676642853,
            "unit": "iter/sec",
            "range": "stddev: 0.00002410616984129453",
            "extra": "mean: 154.2995172421758 usec\nrounds: 29"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 2.5117968725381807,
            "unit": "iter/sec",
            "range": "stddev: 0.000647641836416642",
            "extra": "mean: 398.12136520000365 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 6.153978382959643,
            "unit": "iter/sec",
            "range": "stddev: 0.0011089291363504132",
            "extra": "mean: 162.49650839999674 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 45.400830694339376,
            "unit": "iter/sec",
            "range": "stddev: 0.011794530961575608",
            "extra": "mean: 22.02602870270127 msec\nrounds: 37"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 5.3010490774356,
            "unit": "iter/sec",
            "range": "stddev: 0.0029035909641098585",
            "extra": "mean: 188.64190566667105 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_decompression",
            "value": 2468.50264676917,
            "unit": "iter/sec",
            "range": "stddev: 0.000045012300688880196",
            "extra": "mean: 405.10388000143394 usec\nrounds: 25"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 0.43892968070805605,
            "unit": "iter/sec",
            "range": "stddev: 0.0184233671383794",
            "extra": "mean: 2.2782692626000083 sec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 0.2678616005525004,
            "unit": "iter/sec",
            "range": "stddev: 0.009598598028524339",
            "extra": "mean: 3.733271203999999 sec\nrounds: 5"
          }
        ]
      }
    ]
  }
}