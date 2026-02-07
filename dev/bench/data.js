window.BENCHMARK_DATA = {
  "lastUpdate": 1770430434575,
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
      }
    ]
  }
}