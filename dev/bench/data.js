window.BENCHMARK_DATA = {
  "lastUpdate": 1771720131396,
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
          "message": "Add `SparsityPattern` data structure (#16)\n\n* Add SparsityPattern data structure to replace BCOO in pipeline\n\nReplace BCOO with a custom SparsityPattern class optimized for the\ndetection->coloring->decompression pipeline:\n\n- Store row/col indices separately for direct access (no slicing)\n- Cache col_to_rows mapping for coloring algorithm\n- No unnecessary data values (pattern-only, no all-1s array)\n- Julia-style visualization: dots for small matrices, braille for large\n\nCo-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>\n\n* Update README to showcase SparsityPattern visualization\n\n- Jacobian example: n=100 with braille display\n- Hessian example: n=5 diagonal pattern with dot display\n- Show SparsityPattern pretty-printing output\n\nCo-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>\n\n* Use Julia-style bracket characters for braille frame\n\nBox-drawing characters (\u250c\u2500\u2510\u2502\u2514\u2518) have inconsistent width with braille\nin some fonts. Switch to mathematical bracket characters (\u23a1\u23a2\u23a3\u23a4\u23a5\u23a6)\nwhich align correctly, matching Julia's SparseArrays visualization.\n\nCo-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.5 <noreply@anthropic.com>",
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
          "message": "Improve coloring and API (#21)\n\n* Add LargestFirst ordering, column coloring, and star coloring\n\nRefactor greedy coloring into _greedy_color helper with LargestFirst\nvertex ordering (sort by decreasing degree) for fewer colors. Add\ncolor_cols for column coloring + JVP-based Jacobian computation, with\nauto direction selection in sparse_jacobian. Add star_color for\nsymmetric Hessian coloring (Gebremedhin et al. 2005) with symmetric\ndecompression, used by default in sparse_hessian.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Add SparseMatrixColorings.jl attribution\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Add SMC-sourced test cases for coloring algorithms\n\nPort test matrices from SparseMatrixColorings.jl: Gebremedhin et al.\nFigures 4.1 and 6.1, banded matrices with known star chromatic numbers,\nanti-diagonal, triangle, bidiagonal, and small hand-crafted patterns.\nTighten arrow matrix assertions to exact counts verified against SMC.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Add unified color() function and ColoringResult dataclass\n\nReplace the separate colors/partition parameters on sparse_jacobian\nwith a single coloring: ColoringResult that carries the color array,\ncount, and partition together. The new color(sparsity) function\nauto-picks the best of row/column coloring (ties favor column/JVPs).\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Rename ColoringResult to ColoredPattern and simplify sparse_jacobian API\n\nBundle the sparsity pattern into ColoredPattern so callers pass a single\nobject instead of separate sparsity and coloring arguments.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Add coloring convenience functions, compressed visualization, and rename API\n\n- Add jacobian_coloring() and hessian_coloring() one-step convenience functions\n- Add ColoredPattern._compressed_pattern() and side-by-side/stacked __str__\n- Extract SparsityPattern._render() helper for reuse by ColoredPattern\n- Move ColoredPattern from coloring.py to pattern.py\n- Rename sparse_jacobian \u2192 jacobian, sparse_hessian \u2192 hessian\n- Add colored_pattern parameter to hessian() matching jacobian() API\n- Update README, CLAUDE.md, exports, tests, and pytest markers\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Update README to describe automatic coloring and AD mode selection\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Rename color() to color_jacobian_pattern() and add color_hessian_pattern()\n\nThe generic name `color()` was misleading since it only handled Jacobians.\nThe new names clarify intent and `color_hessian_pattern` wraps star_color\nwith the same nnz==0 early-return guard, simplifying callers.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Fix SIM108 lint warning and add test for size-0 binary elementwise\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Fix printing\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
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
          "id": "89488df953a78b3644812d12be7ffffd3764b987",
          "message": "Batch AD calls via `vmap`, keep decompression on-device\n\nReplace Python for-loops over colors with jax.vmap for VJPs, JVPs,\nand HVPs, keeping all data on-device. Use JAX indexing instead of\nnumpy for decompression. Remove redundant vmap benchmarks from tests.\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-07T21:33:21+01:00",
          "tree_id": "708c2f9402d225a4bbaea12b578c2e8242cb085b",
          "url": "https://github.com/adrhill/asdex/commit/89488df953a78b3644812d12be7ffffd3764b987"
        },
        "date": 1770496450932,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 727.301436412007,
            "unit": "iter/sec",
            "range": "stddev: 0.002791524460996081",
            "extra": "mean: 1.374945723926101 msec\nrounds: 163"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3261.790165840737,
            "unit": "iter/sec",
            "range": "stddev: 0.000007259324502826258",
            "extra": "mean: 306.58011372790037 usec\nrounds: 1996"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_decompression",
            "value": 1144.594309998927,
            "unit": "iter/sec",
            "range": "stddev: 0.00007391143476312659",
            "extra": "mean: 873.6719999952973 usec\nrounds: 7"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 39.242004886886974,
            "unit": "iter/sec",
            "range": "stddev: 0.019707559042545512",
            "extra": "mean: 25.48289780000914 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 82.29673599131378,
            "unit": "iter/sec",
            "range": "stddev: 0.00046944607380491405",
            "extra": "mean: 12.151150199997574 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 21.94765248068138,
            "unit": "iter/sec",
            "range": "stddev: 0.014762750071992986",
            "extra": "mean: 45.56295945000102 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 240.01092003531514,
            "unit": "iter/sec",
            "range": "stddev: 0.00027559362983513004",
            "extra": "mean: 4.166477091345929 msec\nrounds: 208"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_decompression",
            "value": 913.3014660748034,
            "unit": "iter/sec",
            "range": "stddev: 0.00010436475886961596",
            "extra": "mean: 1.0949287142807407 msec\nrounds: 7"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 29.958610142897008,
            "unit": "iter/sec",
            "range": "stddev: 0.0004471235808667465",
            "extra": "mean: 33.37938560000566 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 10.062129139602666,
            "unit": "iter/sec",
            "range": "stddev: 0.026102539081607687",
            "extra": "mean: 99.38254480000523 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 45.78957217379344,
            "unit": "iter/sec",
            "range": "stddev: 0.013244675307261514",
            "extra": "mean: 21.839033485714157 msec\nrounds: 35"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 4.951086520516684,
            "unit": "iter/sec",
            "range": "stddev: 0.0012975086037200852",
            "extra": "mean: 201.97586849999993 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_decompression",
            "value": 827.5514089760512,
            "unit": "iter/sec",
            "range": "stddev: 0.0002740368779836229",
            "extra": "mean: 1.2083841428501987 msec\nrounds: 7"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 23.608438262115232,
            "unit": "iter/sec",
            "range": "stddev: 0.0009079043094998093",
            "extra": "mean: 42.357736199971896 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 0.6443891470776115,
            "unit": "iter/sec",
            "range": "stddev: 0.023783692165777384",
            "extra": "mean: 1.5518572970000037 sec\nrounds: 5"
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
          "id": "8c906fcdc43c840fcf4cc744c472dbfd11099bf6",
          "message": "Remove decompression benchmarks\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-07T21:43:35+01:00",
          "tree_id": "530d325a8edcda1b2859c554208833c7e43a9fa7",
          "url": "https://github.com/adrhill/asdex/commit/8c906fcdc43c840fcf4cc744c472dbfd11099bf6"
        },
        "date": 1770497062600,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 640.3495428431462,
            "unit": "iter/sec",
            "range": "stddev: 0.004719245327127077",
            "extra": "mean: 1.561647089744156 msec\nrounds: 156"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3240.6396371538676,
            "unit": "iter/sec",
            "range": "stddev: 0.000010220936174500636",
            "extra": "mean: 308.58105558391014 usec\nrounds: 1997"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 56.54359263898794,
            "unit": "iter/sec",
            "range": "stddev: 0.0008917314295607732",
            "extra": "mean: 17.68546979999428 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 81.01150183485781,
            "unit": "iter/sec",
            "range": "stddev: 0.0004783743952956105",
            "extra": "mean: 12.343926199991984 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 22.451508104309596,
            "unit": "iter/sec",
            "range": "stddev: 0.01227869954660014",
            "extra": "mean: 44.54043778947966 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 243.22197002117858,
            "unit": "iter/sec",
            "range": "stddev: 0.000039076235597581764",
            "extra": "mean: 4.111470686274455 msec\nrounds: 204"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 29.638098661428163,
            "unit": "iter/sec",
            "range": "stddev: 0.0006911589741707278",
            "extra": "mean: 33.7403560000098 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 11.477285962148608,
            "unit": "iter/sec",
            "range": "stddev: 0.0017688366123536925",
            "extra": "mean: 87.12861239999938 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 50.173253031959945,
            "unit": "iter/sec",
            "range": "stddev: 0.0006052346434612794",
            "extra": "mean: 19.930938090918847 msec\nrounds: 11"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 5.3672843601097595,
            "unit": "iter/sec",
            "range": "stddev: 0.002300898346681258",
            "extra": "mean: 186.31395933334716 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 17.40590769248556,
            "unit": "iter/sec",
            "range": "stddev: 0.03181374174194482",
            "extra": "mean: 57.45175819998849 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 0.6563661491220518,
            "unit": "iter/sec",
            "range": "stddev: 0.011518020236517212",
            "extra": "mean: 1.523539873799996 sec\nrounds: 5"
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
          "id": "9f13bfd1303ff58eeb096d81748ddea232b82ecd",
          "message": "Support `while_loop`, `cond`, and `dynamic_slice` primitives (#24)\n\n* Add brusselator sparsity demo with diffrax\n\nDemonstrates asdex on a realistic reaction-diffusion ODE: detects the\nJacobian sparsity of the brusselator RHS (768 nnz, 11 colors) and\nshows the expected failure on diffrax's `while` primitive.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Support while_loop, cond, dynamic_slice, and diffrax primitives\n\nAdd handlers for 10 new JAX primitives needed by diffrax's diffeqsolve:\n- while: fixed-point iteration over body jaxpr\n- cond: union output deps across all branch jaxprs\n- dynamic_slice / dynamic_update_slice: precise when starts are static\n- not: zero derivative\n- select_if_vmap, nonbatchable, unvmap_any, unvmap_max, pure_callback:\n  conservative fallback\n\nAlso fix select_n to be element-wise instead of globally conservative,\nwhich is necessary for correct sparsity through diffrax control flow.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Simplify dynamic_slice handlers, move select_if_vmap to conservative\n\nReplace manual stride loops with np.indices + np.ravel_multi_index,\nextract _resolve_starts helper. Move select_if_vmap back to the\nconservative fallback since it's a different primitive from select_n.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Fix const_vals propagation into nested jaxprs, clean up handlers\n\nForward const_vals from outer-scope atoms to inner jaxpr variables so\nthat gather/scatter inside cond, while_loop, jit, and custom_jvp can\nresolve static indices precisely instead of falling back to conservative.\n\n- Add seed_const_vals and forward_const_vals helpers to _commons.py\n- Apply both helpers in prop_cond, prop_while, prop_nested_jaxpr,\n  and prop_custom_call\n- Extract prop_broadcast_in_dim into _broadcast.py\n- Clean up select_n: remove dead scalar broadcast, rename branches\u2192cases\n- Add JAX doc URLs to handler docstrings\n- Add scan/associative_scan as explicit TODO errors\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Remove examples/ directory\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-08T00:17:07+01:00",
          "tree_id": "bb07835873f0a87d8bd6df37bd2c719aaf555793",
          "url": "https://github.com/adrhill/asdex/commit/9f13bfd1303ff58eeb096d81748ddea232b82ecd"
        },
        "date": 1770506274960,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 712.9003945119997,
            "unit": "iter/sec",
            "range": "stddev: 0.0029397769552276408",
            "extra": "mean: 1.4027205030297787 msec\nrounds: 165"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3207.5571957888783,
            "unit": "iter/sec",
            "range": "stddev: 0.000007936247832828974",
            "extra": "mean: 311.7637313881339 usec\nrounds: 1988"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 58.33989375160917,
            "unit": "iter/sec",
            "range": "stddev: 0.0004206301673954495",
            "extra": "mean: 17.140929400002847 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 80.17760429766584,
            "unit": "iter/sec",
            "range": "stddev: 0.0003403211083427419",
            "extra": "mean: 12.472310800001196 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 21.871985056219785,
            "unit": "iter/sec",
            "range": "stddev: 0.014465384747426549",
            "extra": "mean: 45.720587200000296 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 243.01367099820655,
            "unit": "iter/sec",
            "range": "stddev: 0.00003484183347459255",
            "extra": "mean: 4.114994830917887 msec\nrounds: 207"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 28.623636284809454,
            "unit": "iter/sec",
            "range": "stddev: 0.0002642951939543367",
            "extra": "mean: 34.936162200003196 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 11.450401888038437,
            "unit": "iter/sec",
            "range": "stddev: 0.0009428151194036761",
            "extra": "mean: 87.33317919999308 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 47.654767640950446,
            "unit": "iter/sec",
            "range": "stddev: 0.010315826024813544",
            "extra": "mean: 20.984259277778644 msec\nrounds: 36"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 5.0968249917397666,
            "unit": "iter/sec",
            "range": "stddev: 0.001562820144485054",
            "extra": "mean: 196.20057616666506 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 17.47956588057925,
            "unit": "iter/sec",
            "range": "stddev: 0.031907677120338165",
            "extra": "mean: 57.20965879999653 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 0.6563157068829131,
            "unit": "iter/sec",
            "range": "stddev: 0.0072790151265666484",
            "extra": "mean: 1.5236569680000058 sec\nrounds: 5"
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
          "id": "4826cb6a0a83177816d1f47b643cd8478fecff5f",
          "message": "Clean up interpreter module and tests (#25)\n\n* Add JAX doc URLs and improve handler docstrings\n\n- Add JAX doc URLs to all handler docstrings missing them\n- Add JAX doc URLs to test module docstrings\n- Fix parameter names to match JAX API (start_indices, scatter_indices)\n- Document precise-path conditions in gather and scatter\n- Document conv assumption (feature_group_count=1, batch_group_count=1)\n- Clarify reduce_sum vs jax.lax.reduce naming difference\n- Document reshape bug with dimensions parameter\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Fix reshape handler ignoring `dimensions` parameter\n\nWhen `dimensions` is not None, `jax.lax.reshape` transposes the input\naxes before reshaping (e.g. `ravel(order='F')` emits `dimensions=(1,0)`).\nThe handler previously passed deps through in the original flat order,\nproducing incorrect (not merely conservative) sparsity patterns.\n\nFix by building a flat index mapping via `np.arange().reshape().transpose()`,\nmirroring the actual element reordering JAX performs.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Add 3D test for reshape dimensions fix\n\nTest ravel(order='F') on a (2, 3, 4) tensor,\nwhich emits dimensions=(2, 1, 0) \u2014 a higher-rank permutation\nthan the 2D case already tested.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Extract shared utilities from `_interpret` handlers\n\nAdd `atom_shape`, `flat_to_coords`, and `conservative_deps` helpers to\n`_commons.py` to reduce duplication across handler modules.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Split `_indexing.py` into `_slice.py`, `_squeeze.py`, and `_reshape.py`\n\nEach handler now lives in its own module, consistent with the rest of\nthe `_interpret` package.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Split test files to match `_interpret` handler modules\n\nOne test file per handler: `_foo.py` \u2192 `test_foo.py`.\nMove `test_control_flow.py` \u2192 `_interpret/test_select.py`,\nrename `test_dynamic_indexing.py` \u2192 `test_dynamic_slice.py`,\nand move fallback/custom_call tests into `test_internals.py`.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-08T01:17:51+01:00",
          "tree_id": "bcca8d345906904d2f614bb3464bd4cc2c7d63e0",
          "url": "https://github.com/adrhill/asdex/commit/4826cb6a0a83177816d1f47b643cd8478fecff5f"
        },
        "date": 1770509917494,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 661.4605313115261,
            "unit": "iter/sec",
            "range": "stddev: 0.003868998829324764",
            "extra": "mean: 1.511806000000071 msec\nrounds: 160"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3274.813967175358,
            "unit": "iter/sec",
            "range": "stddev: 0.000007622603729831648",
            "extra": "mean: 305.3608571428365 usec\nrounds: 2058"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 56.08014669490991,
            "unit": "iter/sec",
            "range": "stddev: 0.0010245683445397706",
            "extra": "mean: 17.831622399995695 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 40.25880096439686,
            "unit": "iter/sec",
            "range": "stddev: 0.02635418907275455",
            "extra": "mean: 24.839289200002668 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 22.378545760733655,
            "unit": "iter/sec",
            "range": "stddev: 0.012485335515297359",
            "extra": "mean: 44.68565610526142 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 243.325166666264,
            "unit": "iter/sec",
            "range": "stddev: 0.000035022010391951884",
            "extra": "mean: 4.1097269702955295 msec\nrounds: 202"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 28.923406261884008,
            "unit": "iter/sec",
            "range": "stddev: 0.0010631299104126302",
            "extra": "mean: 34.57407440000679 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 9.759748849650636,
            "unit": "iter/sec",
            "range": "stddev: 0.02777464658779883",
            "extra": "mean: 102.46165299999461 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 43.62749628088578,
            "unit": "iter/sec",
            "range": "stddev: 0.013650826531622348",
            "extra": "mean: 22.921324514286262 msec\nrounds: 35"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 5.30002260645173,
            "unit": "iter/sec",
            "range": "stddev: 0.0005517831193163254",
            "extra": "mean: 188.67844049998914 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 23.147545561808457,
            "unit": "iter/sec",
            "range": "stddev: 0.0002724181699510639",
            "extra": "mean: 43.20112460000587 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 0.6490283559454185,
            "unit": "iter/sec",
            "range": "stddev: 0.008256847120334209",
            "extra": "mean: 1.5407647305999945 sec\nrounds: 5"
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
          "id": "6aadc3646bb475b13be22acad0a2816c4ed1ab28",
          "message": "Clean up public API and expand linting (#26)\n\n* Reorder modules for readability, rename `star_color` to `color_symmetric`\n\nPut public API functions at the top of coloring.py and decompression.py,\nwith private helpers below. Rename `star_color` to `color_symmetric` for\na consistent naming convention with `color_rows` and `color_cols`.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Make docstrings consistent across jacobian/hessian function pairs\n\nStandardize f parameter descriptions, return value formatting,\nand cross-references across detection, coloring, and decompression.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Update Claude settings\n\n* Simplify pattern.py: extract display, drop `nse` alias and `astype`\n\n- Move ~190 lines of visualization code (dots, braille, compressed\n  pattern rendering) from pattern.py to new _display.py module.\n  pattern.py goes from 561 to 321 lines.\n- Unify on `nnz` as the single property name, drop `nse` alias.\n- Remove `astype()` compatibility shim on SparsityPattern.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Expand ruff lint rules and fix all violations\n\nAdd PT, PIE, ICN, RSE, RET, D, PERF, T20, PLC0415 rules with Google\npydocstyle convention. Fix all violations: docstring formatting (D205,\nD403, D415), list comprehensions (PERF401), top-level imports\n(PLC0415), unnecessary else after return (RET505), pytest style (PT006,\nPT011), and ClassVar annotation (RUF012). Break circular import between\n_cond/_while and __init__ by passing prop_jaxpr as a callback.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-08T02:23:57+01:00",
          "tree_id": "1db64e11b9895efabdb7cc5203904e6ac38c04e2",
          "url": "https://github.com/adrhill/asdex/commit/6aadc3646bb475b13be22acad0a2816c4ed1ab28"
        },
        "date": 1770513887267,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 612.6945735212505,
            "unit": "iter/sec",
            "range": "stddev: 0.005448398002919871",
            "extra": "mean: 1.6321345793106103 msec\nrounds: 145"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3241.7554418963855,
            "unit": "iter/sec",
            "range": "stddev: 0.000009602478330656638",
            "extra": "mean: 308.4748426966511 usec\nrounds: 1958"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 54.8423190096478,
            "unit": "iter/sec",
            "range": "stddev: 0.0005226781676528589",
            "extra": "mean: 18.234094000001733 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 66.13110698290158,
            "unit": "iter/sec",
            "range": "stddev: 0.0005849134963231956",
            "extra": "mean: 15.121476799996003 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 22.131987030739772,
            "unit": "iter/sec",
            "range": "stddev: 0.016345125512621825",
            "extra": "mean: 45.18347126315728 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 237.3843966652847,
            "unit": "iter/sec",
            "range": "stddev: 0.000039951660908164063",
            "extra": "mean: 4.212576791262376 msec\nrounds: 206"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 19.25129184831375,
            "unit": "iter/sec",
            "range": "stddev: 0.03501476786651012",
            "extra": "mean: 51.944566000000236 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 10.815498025498261,
            "unit": "iter/sec",
            "range": "stddev: 0.0003944519437488283",
            "extra": "mean: 92.45991240000535 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 39.98685011264547,
            "unit": "iter/sec",
            "range": "stddev: 0.01887389307953793",
            "extra": "mean: 25.00822138235288 msec\nrounds: 34"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 5.23865329894252,
            "unit": "iter/sec",
            "range": "stddev: 0.0019959349301302237",
            "extra": "mean: 190.88875383332984 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 21.769659827120982,
            "unit": "iter/sec",
            "range": "stddev: 0.0003420712859297489",
            "extra": "mean: 45.935490400000845 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 0.6408715336879672,
            "unit": "iter/sec",
            "range": "stddev: 0.028526118680394126",
            "extra": "mean: 1.560375125799999 sec\nrounds: 5"
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
          "id": "1cdf40724e1df3cab3eec9c7ce1136924c511b79",
          "message": "Update TODOs",
          "timestamp": "2026-02-08T02:31:33+01:00",
          "tree_id": "a559e6bf5951754308d220a50944ea8fd3227d0d",
          "url": "https://github.com/adrhill/asdex/commit/1cdf40724e1df3cab3eec9c7ce1136924c511b79"
        },
        "date": 1770514356131,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 712.6907432546981,
            "unit": "iter/sec",
            "range": "stddev: 0.002843062462979199",
            "extra": "mean: 1.4031331393939892 msec\nrounds: 165"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3270.0731688782103,
            "unit": "iter/sec",
            "range": "stddev: 0.000015082466393729239",
            "extra": "mean: 305.80355495319003 usec\nrounds: 2029"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 60.54302540957044,
            "unit": "iter/sec",
            "range": "stddev: 0.0005413734758628894",
            "extra": "mean: 16.517179199999532 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 82.70956536121912,
            "unit": "iter/sec",
            "range": "stddev: 0.0003642976032704122",
            "extra": "mean: 12.09050000000218 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 21.288665869920912,
            "unit": "iter/sec",
            "range": "stddev: 0.013699066057989978",
            "extra": "mean: 46.97335221052606 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 243.684626458743,
            "unit": "iter/sec",
            "range": "stddev: 0.00003418447881757344",
            "extra": "mean: 4.103664701922856 msec\nrounds: 208"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 30.617978520593457,
            "unit": "iter/sec",
            "range": "stddev: 0.0009777858479013306",
            "extra": "mean: 32.660549400000605 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 10.29548626238459,
            "unit": "iter/sec",
            "range": "stddev: 0.021716916662434395",
            "extra": "mean: 97.12994360000096 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 46.53926525566089,
            "unit": "iter/sec",
            "range": "stddev: 0.012149573734231483",
            "extra": "mean: 21.487232222222573 msec\nrounds: 36"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 5.250599717592818,
            "unit": "iter/sec",
            "range": "stddev: 0.0016071691772904192",
            "extra": "mean: 190.45443450000002 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 24.091312666389808,
            "unit": "iter/sec",
            "range": "stddev: 0.0008468365036023959",
            "extra": "mean: 41.508738599998196 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 0.645964836906171,
            "unit": "iter/sec",
            "range": "stddev: 0.022681246480375273",
            "extra": "mean: 1.5480718807999978 sec\nrounds: 5"
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
          "id": "0ba34afc440e820d41974512e00e2abf9963bfa4",
          "message": "Update TODOs",
          "timestamp": "2026-02-08T02:34:14+01:00",
          "tree_id": "f53f6ae0e0efb95a0647e8b59b79e122e7846ecf",
          "url": "https://github.com/adrhill/asdex/commit/0ba34afc440e820d41974512e00e2abf9963bfa4"
        },
        "date": 1770514548007,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 694.1023545002508,
            "unit": "iter/sec",
            "range": "stddev: 0.003459001506109013",
            "extra": "mean: 1.4407097073168604 msec\nrounds: 164"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3283.288792643737,
            "unit": "iter/sec",
            "range": "stddev: 0.00000657511215361271",
            "extra": "mean: 304.5726596577543 usec\nrounds: 2045"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 60.28633186233281,
            "unit": "iter/sec",
            "range": "stddev: 0.00043161097943263897",
            "extra": "mean: 16.587507800002754 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 80.74478211876517,
            "unit": "iter/sec",
            "range": "stddev: 0.00024056347676489954",
            "extra": "mean: 12.384701199999881 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 21.435274383921353,
            "unit": "iter/sec",
            "range": "stddev: 0.01688006910089219",
            "extra": "mean: 46.65207368421196 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 244.54909670727938,
            "unit": "iter/sec",
            "range": "stddev: 0.000026353397777145783",
            "extra": "mean: 4.089158428571016 msec\nrounds: 210"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 30.163248144096872,
            "unit": "iter/sec",
            "range": "stddev: 0.0008679081956083899",
            "extra": "mean: 33.15292819999911 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 11.21412332528969,
            "unit": "iter/sec",
            "range": "stddev: 0.0015875831170573263",
            "extra": "mean: 89.17326579999667 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 52.631467291941746,
            "unit": "iter/sec",
            "range": "stddev: 0.00042470736968479186",
            "extra": "mean: 19.00004030769454 msec\nrounds: 13"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 5.380295879597647,
            "unit": "iter/sec",
            "range": "stddev: 0.0004908309553788828",
            "extra": "mean: 185.86338416666828 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 23.195135801886472,
            "unit": "iter/sec",
            "range": "stddev: 0.00027909114029105295",
            "extra": "mean: 43.11248739999485 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 0.6386783892062186,
            "unit": "iter/sec",
            "range": "stddev: 0.013664416081184756",
            "extra": "mean: 1.5657332655999994 sec\nrounds: 5"
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
          "id": "709b88779c03a2d3224f064d897e05c31776a249",
          "message": "Add MkDocs documentation site (#27)\n\n* Add MkDocs documentation site\n\nSet up MkDocs with Material theme following the Diataxis framework:\ntutorials, how-to guides, explanations, and auto-generated API reference.\nIncludes GitHub Actions workflow for deployment alongside existing benchmarks.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Improve docs: executable code blocks, mermaid diagrams, content updates\n\n- Add markdown-exec for live code execution in docs\n- Add mermaid diagrams to pipeline explanation page\n- Rewrite getting-started with one-call API, benchmarks, and Hessian example\n- Rename \"The 3-Stage Pipeline\" to \"The ASD Pipeline\"\n- Remove contributing page (user edit)\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Restructure docs: add Tutorials tab, executable examples, and manual sparsity docs\n\n- Move getting-started.md into tutorials/ folder with its own nav tab\n- Remove section overview index.md pages (tutorials, how-to, explanation)\n- Add executable code blocks with hidden print cells to how-to guides\n- Document manual SparsityPattern construction (from_dense, from_coordinates, from_bcoo)\n- Add precompute warning admonitions to one-call API sections\n- Expand docs/CLAUDE.md with structure overview, clearer Diataxis descriptions, and restart note\n- Update mkdocs.yml nav and fix cross-section links\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Improve Hessian how-to guide structure, gitignore site/, and polish tutorial\n\n- Restructure hessians.md to match jacobians.md layout (separate sections for\n  manual sparsity, detect+color, precomputing)\n- Add executable code blocks with hidden prints to hessians.md\n- Use sparse multi-dimensional example (sum(x**3)) instead of dense one\n- Add site/ to .gitignore\n- Minor tutorial polish\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Split API reference into per-topic pages and remove pipeline explanation\n\n- Remove explanation/pipeline.md\n- Split reference/api.md into jacobian.md, hessian.md, sparsity.md,\n  coloring.md, and data-structures.md\n- Keep reference/api.md as a full API page with all docstrings\n- Update nav, docs/CLAUDE.md structure tree, and cross-links\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Add custom CSS for sidebar section headers and tweak reference pages\n\n- Style sidebar section headers as small, bold, uppercase labels\n  to visually distinguish them from page links\n- Move color_jacobian_pattern/color_hessian_pattern back to coloring reference\n- Swap order of data structures (ColoredPattern first)\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Add sparsity detection explanation page\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Update the Getting started guide.\n\n* Trim README\n\n* Fix API reference\n\n* Expand the graph coloring explanation page\n\nAdd concrete Jacobian example, compression/decompression section,\nand references. Streamline symmetric coloring discussion.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-08T16:55:30+01:00",
          "tree_id": "0579cbee5577e6e920b7f81d5bc2f01c23d7bba1",
          "url": "https://github.com/adrhill/asdex/commit/709b88779c03a2d3224f064d897e05c31776a249"
        },
        "date": 1770566178780,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 723.1767987559738,
            "unit": "iter/sec",
            "range": "stddev: 0.002757190643072523",
            "extra": "mean: 1.3827877245512081 msec\nrounds: 167"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3198.8332419460407,
            "unit": "iter/sec",
            "range": "stddev: 0.000007690668532322336",
            "extra": "mean: 312.6139827756825 usec\nrounds: 2032"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 61.48627350301963,
            "unit": "iter/sec",
            "range": "stddev: 0.00045187093142807697",
            "extra": "mean: 16.263792600000215 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 83.36331633943327,
            "unit": "iter/sec",
            "range": "stddev: 0.0003361716729393651",
            "extra": "mean: 11.99568400000146 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 22.444171310600588,
            "unit": "iter/sec",
            "range": "stddev: 0.012198139784861858",
            "extra": "mean: 44.554997649999706 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 239.40403633936577,
            "unit": "iter/sec",
            "range": "stddev: 0.000024446120423793942",
            "extra": "mean: 4.177039014423533 msec\nrounds: 208"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 30.961574741357943,
            "unit": "iter/sec",
            "range": "stddev: 0.0010937023342586971",
            "extra": "mean: 32.298098800001185 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 11.507784520242707,
            "unit": "iter/sec",
            "range": "stddev: 0.0015505797816357453",
            "extra": "mean: 86.89769939999792 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 46.879095267421185,
            "unit": "iter/sec",
            "range": "stddev: 0.011706003566131759",
            "extra": "mean: 21.331469694445104 msec\nrounds: 36"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 5.0739246208727735,
            "unit": "iter/sec",
            "range": "stddev: 0.0006657095718346416",
            "extra": "mean: 197.08609699999613 msec\nrounds: 6"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 22.886192156990347,
            "unit": "iter/sec",
            "range": "stddev: 0.0006941844016833498",
            "extra": "mean: 43.694468400002506 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 0.6523400235172981,
            "unit": "iter/sec",
            "range": "stddev: 0.012243624098424461",
            "extra": "mean: 1.5329428886000016 sec\nrounds: 5"
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
          "id": "f4fffeb04244ff84d63c4d73800a96aaef9dd657",
          "message": "Add precise `pad` primitive handler (#28)\n\n* Add precise `pad` primitive handler\n\nReplaces the conservative fallback with a handler that reverse-maps each\noutput element to its input source, correctly handling low/high padding,\nnegative padding (trimming), and interior padding (dilation).\n\nThis fixes Hessian sparsity for functions using sliced operations\n(e.g. finite differences), where JAX's grad emits `pad` primitives.\nFor example, `sum((x[1:]-x[:-1])^2)` now gives the correct 13-nnz\ntridiagonal Hessian instead of a fully dense 25-nnz pattern.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Update project metadata in pyproject.toml\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-08T20:43:50+01:00",
          "tree_id": "e8a3cd6e5d415c5d0462305804b0690d222e2990",
          "url": "https://github.com/adrhill/asdex/commit/f4fffeb04244ff84d63c4d73800a96aaef9dd657"
        },
        "date": 1770579863461,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 697.5422604595892,
            "unit": "iter/sec",
            "range": "stddev: 0.0032393256818053326",
            "extra": "mean: 1.4336048963415216 msec\nrounds: 164"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3233.9243486365335,
            "unit": "iter/sec",
            "range": "stddev: 0.000006546585548055818",
            "extra": "mean: 309.22182840226725 usec\nrounds: 2028"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 59.95912993834298,
            "unit": "iter/sec",
            "range": "stddev: 0.000497500155944088",
            "extra": "mean: 16.67802719999969 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 81.12605956108412,
            "unit": "iter/sec",
            "range": "stddev: 0.0003830124252539208",
            "extra": "mean: 12.326495400002102 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 23.550483462811844,
            "unit": "iter/sec",
            "range": "stddev: 0.008643191556736669",
            "extra": "mean: 42.461973299999656 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 242.95387575596135,
            "unit": "iter/sec",
            "range": "stddev: 0.000026421486727964214",
            "extra": "mean: 4.116007603864961 msec\nrounds: 207"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 23.93998739281899,
            "unit": "iter/sec",
            "range": "stddev: 0.019302575564457542",
            "extra": "mean: 41.77111640000106 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 11.273706392078129,
            "unit": "iter/sec",
            "range": "stddev: 0.0023934340812666946",
            "extra": "mean: 88.70197299999631 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 88.83794533076049,
            "unit": "iter/sec",
            "range": "stddev: 0.012183160993606203",
            "extra": "mean: 11.256451241379015 msec\nrounds: 58"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3238.915607738146,
            "unit": "iter/sec",
            "range": "stddev: 0.000009191702282507735",
            "extra": "mean: 308.74530895799927 usec\nrounds: 2188"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 24.792111820396922,
            "unit": "iter/sec",
            "range": "stddev: 0.0014094636234731404",
            "extra": "mean: 40.335410200000865 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 19.012303892316236,
            "unit": "iter/sec",
            "range": "stddev: 0.011866928769555456",
            "extra": "mean: 52.597518199998206 msec\nrounds: 20"
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
          "id": "e3ae944fb70fbc30c69a3de0300327d88b7c8f96",
          "message": "Docs: use Rosenbrock example in Hessian How-to\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-08T20:54:59+01:00",
          "tree_id": "814807c3bf5f9c460272cf4085dc0d754a21af49",
          "url": "https://github.com/adrhill/asdex/commit/e3ae944fb70fbc30c69a3de0300327d88b7c8f96"
        },
        "date": 1770580538117,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 656.8824266643404,
            "unit": "iter/sec",
            "range": "stddev: 0.004356284314289112",
            "extra": "mean: 1.5223424457829025 msec\nrounds: 166"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3235.0562435182023,
            "unit": "iter/sec",
            "range": "stddev: 0.000011401429372377512",
            "extra": "mean: 309.11363658780647 usec\nrounds: 2028"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 59.77068808160453,
            "unit": "iter/sec",
            "range": "stddev: 0.0005521331337005912",
            "extra": "mean: 16.730608799997526 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 81.23589298100775,
            "unit": "iter/sec",
            "range": "stddev: 0.00032859974012539435",
            "extra": "mean: 12.309829599999489 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 21.184254099399144,
            "unit": "iter/sec",
            "range": "stddev: 0.015734912948974146",
            "extra": "mean: 47.20487185000124 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 240.86002415603278,
            "unit": "iter/sec",
            "range": "stddev: 0.00009739204979430773",
            "extra": "mean: 4.151789004854475 msec\nrounds: 206"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 29.560585910674885,
            "unit": "iter/sec",
            "range": "stddev: 0.0009355068356940787",
            "extra": "mean: 33.82882880000295 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 11.398375503523575,
            "unit": "iter/sec",
            "range": "stddev: 0.0008333071314144641",
            "extra": "mean: 87.7317999999974 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 89.21798401914732,
            "unit": "iter/sec",
            "range": "stddev: 0.012858542948533544",
            "extra": "mean: 11.208502534482141 msec\nrounds: 58"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3193.6488062075427,
            "unit": "iter/sec",
            "range": "stddev: 0.000007983126948381369",
            "extra": "mean: 313.12146722466326 usec\nrounds: 2151"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 25.20606944810584,
            "unit": "iter/sec",
            "range": "stddev: 0.0006486529175009562",
            "extra": "mean: 39.67298439999922 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 18.766603159825255,
            "unit": "iter/sec",
            "range": "stddev: 0.012681069918010959",
            "extra": "mean: 53.28614834999854 msec\nrounds: 20"
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
          "id": "2fad0d1e9176f840f75b880c962f6d8e5c38375f",
          "message": "Add precise `transpose` primitive handler (#29)\n\n* Add precise `transpose` primitive handler\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Add tests for `transpose` primitive handler\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Add `/add-handler` skill for new primitive handlers\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-08T21:48:54+01:00",
          "tree_id": "643ff1683ef75e3d8e7d4d3fa50b8214b01c5317",
          "url": "https://github.com/adrhill/asdex/commit/2fad0d1e9176f840f75b880c962f6d8e5c38375f"
        },
        "date": 1770583770925,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 627.2783058650452,
            "unit": "iter/sec",
            "range": "stddev: 0.0046852126152313045",
            "extra": "mean: 1.5941887207799328 msec\nrounds: 154"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3236.4013890556494,
            "unit": "iter/sec",
            "range": "stddev: 0.00000887699830795827",
            "extra": "mean: 308.98515968434634 usec\nrounds: 2029"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 57.17520850940324,
            "unit": "iter/sec",
            "range": "stddev: 0.00044086822577100993",
            "extra": "mean: 17.490098000001808 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 79.45349092184229,
            "unit": "iter/sec",
            "range": "stddev: 0.0007122958802572632",
            "extra": "mean: 12.58597939999504 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 20.873501365547583,
            "unit": "iter/sec",
            "range": "stddev: 0.018059539214148643",
            "extra": "mean: 47.90763094736629 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 241.7553895811917,
            "unit": "iter/sec",
            "range": "stddev: 0.000024419997101398102",
            "extra": "mean: 4.136412436274384 msec\nrounds: 204"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 20.319049223247458,
            "unit": "iter/sec",
            "range": "stddev: 0.03233823186424366",
            "extra": "mean: 49.21490119999703 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 11.011243221486124,
            "unit": "iter/sec",
            "range": "stddev: 0.00037835694302140857",
            "extra": "mean: 90.81626660000666 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 81.09136295837334,
            "unit": "iter/sec",
            "range": "stddev: 0.01634039089959695",
            "extra": "mean: 12.33176954385796 msec\nrounds: 57"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3226.416886307873,
            "unit": "iter/sec",
            "range": "stddev: 0.000010074788975693353",
            "extra": "mean: 309.9413483247488 usec\nrounds: 2090"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 24.456390565154894,
            "unit": "iter/sec",
            "range": "stddev: 0.0013220869302057214",
            "extra": "mean: 40.889108199996826 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 17.947357431648637,
            "unit": "iter/sec",
            "range": "stddev: 0.016950621194069877",
            "extra": "mean: 55.71850919047198 msec\nrounds: 21"
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
          "id": "d1b7e377204a3c50537d848b7f8411b4c40a0819",
          "message": "Add precise `rev` primitive handler (#30)\n\n* Add precise `rev` primitive handler\n\nReplace the conservative fallback for `rev` (reverse) with a precise\nhandler that tracks the per-element permutation along reversed dimensions.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Vectorize `prop_rev` and improve add-handler skill\n\nReplace per-element Python loop in `_rev.py` with `np.flip` on an\nindex array.  Update the add-handler skill with detailed adversarial\ntest guidance (step 7) and a new simplification step (step 8).\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-08T22:06:35+01:00",
          "tree_id": "8f6ed223b04ee6c4a5d6578cfd68d8c69b7544c2",
          "url": "https://github.com/adrhill/asdex/commit/d1b7e377204a3c50537d848b7f8411b4c40a0819"
        },
        "date": 1770584829554,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 688.062072190994,
            "unit": "iter/sec",
            "range": "stddev: 0.003341266436772559",
            "extra": "mean: 1.4533572484466453 msec\nrounds: 161"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3266.6485602061334,
            "unit": "iter/sec",
            "range": "stddev: 0.000006893697830328382",
            "extra": "mean: 306.1241457626827 usec\nrounds: 2065"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 60.47664864153817,
            "unit": "iter/sec",
            "range": "stddev: 0.0005048331997152025",
            "extra": "mean: 16.53530779999528 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 81.39657243590915,
            "unit": "iter/sec",
            "range": "stddev: 0.0006694529387191353",
            "extra": "mean: 12.285529600001155 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 21.817267270073398,
            "unit": "iter/sec",
            "range": "stddev: 0.014135430470461764",
            "extra": "mean: 45.8352545999972 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 237.88885136984004,
            "unit": "iter/sec",
            "range": "stddev: 0.000024086860476311063",
            "extra": "mean: 4.203643820387884 msec\nrounds: 206"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 30.84008990873228,
            "unit": "iter/sec",
            "range": "stddev: 0.0009300327449467559",
            "extra": "mean: 32.42532699999856 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 11.40471622925195,
            "unit": "iter/sec",
            "range": "stddev: 0.0010717385893134485",
            "extra": "mean: 87.68302340001242 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 93.96543990937641,
            "unit": "iter/sec",
            "range": "stddev: 0.011526973949098582",
            "extra": "mean: 10.64221059321848 msec\nrounds: 59"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3207.7035312665535,
            "unit": "iter/sec",
            "range": "stddev: 0.000006946108648670315",
            "extra": "mean: 311.7495087225697 usec\nrounds: 2121"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 24.970894799000565,
            "unit": "iter/sec",
            "range": "stddev: 0.0013706872503722798",
            "extra": "mean: 40.046622600004866 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 18.722099998943648,
            "unit": "iter/sec",
            "range": "stddev: 0.01449612935707989",
            "extra": "mean: 53.41281160000335 msec\nrounds: 20"
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
          "id": "7a3887d837973c22afaf1641e1f3b36201bb8cb7",
          "message": "Add precise `dot_general` primitive handler (#31)\n\n* Add precise `dot_general` primitive handler\n\nReplace the conservative fallback for dot_general with a precise handler\nthat tracks per-element dependencies through batch, contracting, and free\ndimensions of generalized matrix multiplications.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Clean up TODOs\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-08T22:24:53+01:00",
          "tree_id": "7c738b1db11b337fff38cf52f921d98889c167a4",
          "url": "https://github.com/adrhill/asdex/commit/7a3887d837973c22afaf1641e1f3b36201bb8cb7"
        },
        "date": 1770585929537,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 721.949824419193,
            "unit": "iter/sec",
            "range": "stddev: 0.0028176224725377542",
            "extra": "mean: 1.3851378117648243 msec\nrounds: 170"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3254.8889196749687,
            "unit": "iter/sec",
            "range": "stddev: 0.00000672363262213771",
            "extra": "mean: 307.23014661276346 usec\nrounds: 1978"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 61.08618170461724,
            "unit": "iter/sec",
            "range": "stddev: 0.00044032042771823915",
            "extra": "mean: 16.370314399998165 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 83.20000947813111,
            "unit": "iter/sec",
            "range": "stddev: 0.00026824467831327475",
            "extra": "mean: 12.019229400002018 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 22.1320355054603,
            "unit": "iter/sec",
            "range": "stddev: 0.013144622159524619",
            "extra": "mean: 45.18337229999858 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 243.97785113300526,
            "unit": "iter/sec",
            "range": "stddev: 0.000029756373135880102",
            "extra": "mean: 4.098732714285802 msec\nrounds: 210"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 31.06508430901801,
            "unit": "iter/sec",
            "range": "stddev: 0.0007681014319213438",
            "extra": "mean: 32.190480800005616 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 11.544441683578457,
            "unit": "iter/sec",
            "range": "stddev: 0.0012331872676036783",
            "extra": "mean: 86.62177239999949 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 97.2132077878127,
            "unit": "iter/sec",
            "range": "stddev: 0.010822953475703495",
            "extra": "mean: 10.286668064515473 msec\nrounds: 62"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3147.8071043122823,
            "unit": "iter/sec",
            "range": "stddev: 0.00002140317040580055",
            "extra": "mean: 317.6814737567839 usec\nrounds: 2172"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 25.614666047944915,
            "unit": "iter/sec",
            "range": "stddev: 0.0012183995004755167",
            "extra": "mean: 39.04013420000183 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 19.05463142851476,
            "unit": "iter/sec",
            "range": "stddev: 0.01258777165146568",
            "extra": "mean: 52.48067923809464 msec\nrounds: 21"
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
          "id": "b8bca5034576ff130f6f22a982caf095f01de617",
          "message": "Add thorough tests for `reshape` primitive handler (#32)\n\nExpand test_reshape.py from 4 to 25 tests covering identity reshapes,\ndimensions param permutations, size-1 dims, chained reshapes,\nnon-contiguous inputs, high-level functions, and compositions.\nUpdate TODO.md to remove outdated entries.\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-08T23:14:22+01:00",
          "tree_id": "a93714cb8e1b49cf248350fb5f84b416f4362fcc",
          "url": "https://github.com/adrhill/asdex/commit/b8bca5034576ff130f6f22a982caf095f01de617"
        },
        "date": 1770588894705,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 706.6814728854604,
            "unit": "iter/sec",
            "range": "stddev: 0.00275834972261733",
            "extra": "mean: 1.41506469090931 msec\nrounds: 165"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3210.5284238577497,
            "unit": "iter/sec",
            "range": "stddev: 0.00001555082009860901",
            "extra": "mean: 311.47520531788547 usec\nrounds: 2031"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 59.8414837394148,
            "unit": "iter/sec",
            "range": "stddev: 0.0005442975279964836",
            "extra": "mean: 16.710815600004025 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 81.36858708735289,
            "unit": "iter/sec",
            "range": "stddev: 0.00039097645122457096",
            "extra": "mean: 12.28975500000331 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 21.99785972171242,
            "unit": "iter/sec",
            "range": "stddev: 0.014321652543305398",
            "extra": "mean: 45.45896794736698 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 239.36885609068517,
            "unit": "iter/sec",
            "range": "stddev: 0.000025899599836146022",
            "extra": "mean: 4.1776529174754 msec\nrounds: 206"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 30.294610604526806,
            "unit": "iter/sec",
            "range": "stddev: 0.0010932768675020377",
            "extra": "mean: 33.00917159999983 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 11.299659591427007,
            "unit": "iter/sec",
            "range": "stddev: 0.000491268843826839",
            "extra": "mean: 88.49824119999994 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 125.73370669657999,
            "unit": "iter/sec",
            "range": "stddev: 0.00016509435429521685",
            "extra": "mean: 7.953316785714395 msec\nrounds: 14"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3218.342954701133,
            "unit": "iter/sec",
            "range": "stddev: 0.000011137192121113374",
            "extra": "mean: 310.7189053731111 usec\nrounds: 2103"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 25.039462067021642,
            "unit": "iter/sec",
            "range": "stddev: 0.0009776746083524448",
            "extra": "mean: 39.936960199997884 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 17.915778143726822,
            "unit": "iter/sec",
            "range": "stddev: 0.01666804526993325",
            "extra": "mean: 55.81672155000135 msec\nrounds: 20"
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
          "id": "faef7a288b68572f0fd1c95936b834140967e993",
          "message": "Add precise handlers for all reduction primitives (#33)\n\n* Add precise `reduce_max` primitive handler\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Add precise `reduce_prod` primitive handler\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Add precise `reduce_min` primitive handler\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Add zero-derivative handlers for `reduce_and`, `reduce_or`, `reduce_xor`\n\nBitwise reductions have zero Jacobian, so they use the existing\n`prop_zero_derivative` \u2014 no separate handler files needed.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Consolidate reduction handlers into single `_reduce.py` module\n\nThe four reduction primitives (reduce_sum, reduce_max, reduce_min,\nreduce_prod) share identical sparsity structure, so they now share\na single `prop_reduce` handler. Tests are parametrized over the\nreduce function and shape/axes combinations.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Parametrize bitwise reduction tests over reduce_and/or/xor\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Update SKILL and TODOs\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-08T23:23:59+01:00",
          "tree_id": "984e3a6736257a3c4644e001b388e6e3c720fbd6",
          "url": "https://github.com/adrhill/asdex/commit/faef7a288b68572f0fd1c95936b834140967e993"
        },
        "date": 1770589473351,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 663.8023996695964,
            "unit": "iter/sec",
            "range": "stddev: 0.00405260089833369",
            "extra": "mean: 1.5064724088038004 msec\nrounds: 159"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3239.075439673935,
            "unit": "iter/sec",
            "range": "stddev: 0.0000065504970684697765",
            "extra": "mean: 308.73007394377515 usec\nrounds: 1988"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 57.2030569862779,
            "unit": "iter/sec",
            "range": "stddev: 0.0007613997673490583",
            "extra": "mean: 17.481583200000728 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 77.37671038705516,
            "unit": "iter/sec",
            "range": "stddev: 0.00026495798700692825",
            "extra": "mean: 12.923785399996746 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 20.776023151863665,
            "unit": "iter/sec",
            "range": "stddev: 0.019496801653107812",
            "extra": "mean: 48.13240689473805 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 242.95915897840848,
            "unit": "iter/sec",
            "range": "stddev: 0.000035858621969400084",
            "extra": "mean: 4.115918099999963 msec\nrounds: 210"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 28.72498049889885,
            "unit": "iter/sec",
            "range": "stddev: 0.0007768260677243277",
            "extra": "mean: 34.812904399998956 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 11.085312804569499,
            "unit": "iter/sec",
            "range": "stddev: 0.0009905455258238083",
            "extra": "mean: 90.20945260000133 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 123.79472041720724,
            "unit": "iter/sec",
            "range": "stddev: 0.00018826480371296914",
            "extra": "mean: 8.077888916666609 msec\nrounds: 12"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3102.2420231733677,
            "unit": "iter/sec",
            "range": "stddev: 0.0000534793497042519",
            "extra": "mean: 322.34751271181375 usec\nrounds: 2124"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 24.786655063206233,
            "unit": "iter/sec",
            "range": "stddev: 0.0017211592353249784",
            "extra": "mean: 40.34429000000159 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 19.42817061122612,
            "unit": "iter/sec",
            "range": "stddev: 0.001038009636240581",
            "extra": "mean: 51.47165011111099 msec\nrounds: 9"
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
          "id": "c55805a539c8645fcdbcc0c7aebd74e29162b6fb",
          "message": "Update README",
          "timestamp": "2026-02-08T23:45:23+01:00",
          "tree_id": "3cb74fbb0ca19e3325dfdab84a6ec1e2226da9df",
          "url": "https://github.com/adrhill/asdex/commit/c55805a539c8645fcdbcc0c7aebd74e29162b6fb"
        },
        "date": 1770590756500,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 724.5264163270951,
            "unit": "iter/sec",
            "range": "stddev: 0.0033559206792397305",
            "extra": "mean: 1.3802119252868474 msec\nrounds: 174"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3417.4198073177645,
            "unit": "iter/sec",
            "range": "stddev: 0.0002142352700067656",
            "extra": "mean: 292.6184245373329 usec\nrounds: 2054"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 65.2019058516735,
            "unit": "iter/sec",
            "range": "stddev: 0.0004551153652242098",
            "extra": "mean: 15.336975000008124 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 82.53170017842771,
            "unit": "iter/sec",
            "range": "stddev: 0.0019781483005266347",
            "extra": "mean: 12.11655640000231 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 24.68647945823764,
            "unit": "iter/sec",
            "range": "stddev: 0.0014967385600157062",
            "extra": "mean: 40.50800365000242 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 264.6384066970842,
            "unit": "iter/sec",
            "range": "stddev: 0.000024586418878405413",
            "extra": "mean: 3.778741009216551 msec\nrounds: 217"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 30.64875532155554,
            "unit": "iter/sec",
            "range": "stddev: 0.0008522561011637792",
            "extra": "mean: 32.627752400003374 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 12.497271783085472,
            "unit": "iter/sec",
            "range": "stddev: 0.0009078759042344923",
            "extra": "mean: 80.0174643999867 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 92.22536269465958,
            "unit": "iter/sec",
            "range": "stddev: 0.013022589038251285",
            "extra": "mean: 10.843004253730154 msec\nrounds: 67"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3460.943119870303,
            "unit": "iter/sec",
            "range": "stddev: 0.0000043635505857461",
            "extra": "mean: 288.938582740266 usec\nrounds: 2248"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 29.085065171548845,
            "unit": "iter/sec",
            "range": "stddev: 0.0005627823384392886",
            "extra": "mean: 34.38190679999593 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 21.410205847082395,
            "unit": "iter/sec",
            "range": "stddev: 0.013586233194707512",
            "extra": "mean: 46.70669713043752 msec\nrounds: 23"
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
          "id": "4bd231ae9d40e1bc5f2e6bca63cdda8b672d02dc",
          "message": "docs: add Brusselator PDE how-to page (#34)\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-09T13:36:13+01:00",
          "tree_id": "b8b8b773bd5f7537740a8489ca504621aee8c29d",
          "url": "https://github.com/adrhill/asdex/commit/4bd231ae9d40e1bc5f2e6bca63cdda8b672d02dc"
        },
        "date": 1770640607310,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 702.9768406410767,
            "unit": "iter/sec",
            "range": "stddev: 0.0030694296715492557",
            "extra": "mean: 1.4225219696968314 msec\nrounds: 165"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3265.929058375829,
            "unit": "iter/sec",
            "range": "stddev: 0.000011803480736036732",
            "extra": "mean: 306.1915865671949 usec\nrounds: 2010"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 57.892165698679335,
            "unit": "iter/sec",
            "range": "stddev: 0.0006949599381724919",
            "extra": "mean: 17.273494399999834 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 80.58182008063119,
            "unit": "iter/sec",
            "range": "stddev: 0.0004395750268183218",
            "extra": "mean: 12.409746999998106 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 21.873975457167514,
            "unit": "iter/sec",
            "range": "stddev: 0.01386214872957634",
            "extra": "mean: 45.71642690000033 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 224.18660175246134,
            "unit": "iter/sec",
            "range": "stddev: 0.0008591072551944504",
            "extra": "mean: 4.460569865384567 msec\nrounds: 208"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 30.05665770146733,
            "unit": "iter/sec",
            "range": "stddev: 0.001011004708438243",
            "extra": "mean: 33.27049899999963 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 11.263280432588632,
            "unit": "iter/sec",
            "range": "stddev: 0.001202995618289872",
            "extra": "mean: 88.78408079999929 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 94.57867191689662,
            "unit": "iter/sec",
            "range": "stddev: 0.011294974110691617",
            "extra": "mean: 10.573208311475014 msec\nrounds: 61"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3224.2280377520324,
            "unit": "iter/sec",
            "range": "stddev: 0.00000966464352875787",
            "extra": "mean: 310.1517598293733 usec\nrounds: 2111"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 25.8706951919908,
            "unit": "iter/sec",
            "range": "stddev: 0.0009797297084456064",
            "extra": "mean: 38.653773800001545 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 18.61011801170359,
            "unit": "iter/sec",
            "range": "stddev: 0.013845812577498744",
            "extra": "mean: 53.734210571427695 msec\nrounds: 21"
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
          "id": "fdd1c6c2527b11ed5fb8fe0eb25e95486c790655",
          "message": "feat!: switch to functional API (#35)\n\n* feat!: switch to functional API\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* refactor: rename returned closures to jac_fn / hess_fn\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Minor tweaks\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-09T16:39:01+01:00",
          "tree_id": "f5a687e1310b7a3e6232ceb990a5b4c331bed0af",
          "url": "https://github.com/adrhill/asdex/commit/fdd1c6c2527b11ed5fb8fe0eb25e95486c790655"
        },
        "date": 1770651579278,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 690.6907152366679,
            "unit": "iter/sec",
            "range": "stddev: 0.0034354438004304878",
            "extra": "mean: 1.447826035503237 msec\nrounds: 169"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3250.5414913495024,
            "unit": "iter/sec",
            "range": "stddev: 0.000007300839078000627",
            "extra": "mean: 307.6410507791542 usec\nrounds: 1989"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 61.15583525679842,
            "unit": "iter/sec",
            "range": "stddev: 0.0004754775896584963",
            "extra": "mean: 16.35166939999948 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 82.8207392141831,
            "unit": "iter/sec",
            "range": "stddev: 0.00032558572711124777",
            "extra": "mean: 12.074270400000842 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 24.391676110269447,
            "unit": "iter/sec",
            "range": "stddev: 0.000397813980269722",
            "extra": "mean: 40.99759260000084 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 241.41475115351932,
            "unit": "iter/sec",
            "range": "stddev: 0.00003322362381768263",
            "extra": "mean: 4.1422489521532375 msec\nrounds: 209"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 30.54673530452859,
            "unit": "iter/sec",
            "range": "stddev: 0.0007840246433365911",
            "extra": "mean: 32.73672260000069 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 11.64007868218244,
            "unit": "iter/sec",
            "range": "stddev: 0.000969405930030996",
            "extra": "mean: 85.91007220000222 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 92.74963893363639,
            "unit": "iter/sec",
            "range": "stddev: 0.013051003496385981",
            "extra": "mean: 10.78171313114775 msec\nrounds: 61"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3187.335942598409,
            "unit": "iter/sec",
            "range": "stddev: 0.000008791022522623594",
            "extra": "mean: 313.74163816091846 usec\nrounds: 2175"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 26.066211534625285,
            "unit": "iter/sec",
            "range": "stddev: 0.0014928111163875154",
            "extra": "mean: 38.36384120000105 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 19.76281618707656,
            "unit": "iter/sec",
            "range": "stddev: 0.01431221401884499",
            "extra": "mean: 50.60007594737065 msec\nrounds: 19"
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
          "id": "3f8fad1869dc0d6cca9d4b79891f6f7d4bb6ee6d",
          "message": "docs: add acknowledgements",
          "timestamp": "2026-02-09T17:01:20+01:00",
          "tree_id": "69e931f164cf4341aae0fe755fadd352ad3b645b",
          "url": "https://github.com/adrhill/asdex/commit/3f8fad1869dc0d6cca9d4b79891f6f7d4bb6ee6d"
        },
        "date": 1770652915672,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 690.7146165524471,
            "unit": "iter/sec",
            "range": "stddev: 0.0033168544132346897",
            "extra": "mean: 1.4477759352933404 msec\nrounds: 170"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3255.1562447542483,
            "unit": "iter/sec",
            "range": "stddev: 0.000009086257618802929",
            "extra": "mean: 307.20491577371155 usec\nrounds: 1959"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 59.78769961829452,
            "unit": "iter/sec",
            "range": "stddev: 0.00023672338029550737",
            "extra": "mean: 16.725848399994447 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 82.11863718937268,
            "unit": "iter/sec",
            "range": "stddev: 0.00030809339140293074",
            "extra": "mean: 12.177503599991724 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 22.666926401011754,
            "unit": "iter/sec",
            "range": "stddev: 0.011099801976726564",
            "extra": "mean: 44.11714152631493 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 239.8267499046422,
            "unit": "iter/sec",
            "range": "stddev: 0.00004275722683018239",
            "extra": "mean: 4.1696766536577385 msec\nrounds: 205"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 21.31054480279828,
            "unit": "iter/sec",
            "range": "stddev: 0.027003823010725823",
            "extra": "mean: 46.92512600000214 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 11.210848801165703,
            "unit": "iter/sec",
            "range": "stddev: 0.000568256521969903",
            "extra": "mean: 89.19931199999951 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 83.74341947148179,
            "unit": "iter/sec",
            "range": "stddev: 0.01510216505124126",
            "extra": "mean: 11.941236771929796 msec\nrounds: 57"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3232.8468006097764,
            "unit": "iter/sec",
            "range": "stddev: 0.000011943892223007663",
            "extra": "mean: 309.3248958816672 usec\nrounds: 2161"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 24.57539541098198,
            "unit": "iter/sec",
            "range": "stddev: 0.0011113620270840575",
            "extra": "mean: 40.691105200005495 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 18.97252283889614,
            "unit": "iter/sec",
            "range": "stddev: 0.014506039165353698",
            "extra": "mean: 52.70780319999773 msec\nrounds: 20"
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
          "id": "023d1c9221e867e4f619ad54c3ba5214a7a2b590",
          "message": "docs: link to GitHub profiles",
          "timestamp": "2026-02-09T17:07:01+01:00",
          "tree_id": "7b7081bad612bd3d8c72349ec77dbb0391c644b6",
          "url": "https://github.com/adrhill/asdex/commit/023d1c9221e867e4f619ad54c3ba5214a7a2b590"
        },
        "date": 1770653266598,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 673.927732539104,
            "unit": "iter/sec",
            "range": "stddev: 0.003648992710532731",
            "extra": "mean: 1.4838386249997153 msec\nrounds: 168"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3179.0926706922714,
            "unit": "iter/sec",
            "range": "stddev: 0.00004418683749074385",
            "extra": "mean: 314.55515884104204 usec\nrounds: 2002"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 57.34550764695828,
            "unit": "iter/sec",
            "range": "stddev: 0.002226169274872178",
            "extra": "mean: 17.438157600005866 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 81.29554532114807,
            "unit": "iter/sec",
            "range": "stddev: 0.0003296518066431856",
            "extra": "mean: 12.300797000003172 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 22.754610322974447,
            "unit": "iter/sec",
            "range": "stddev: 0.01235191156389289",
            "extra": "mean: 43.94713800000076 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 241.52453329802051,
            "unit": "iter/sec",
            "range": "stddev: 0.00002620593230912173",
            "extra": "mean: 4.140366141463923 msec\nrounds: 205"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 29.863865509350177,
            "unit": "iter/sec",
            "range": "stddev: 0.0007383440803365442",
            "extra": "mean: 33.485283399997456 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 11.51133275399558,
            "unit": "iter/sec",
            "range": "stddev: 0.0008788092802154955",
            "extra": "mean: 86.87091420000002 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 89.48608494883781,
            "unit": "iter/sec",
            "range": "stddev: 0.013904654006338902",
            "extra": "mean: 11.174921783333502 msec\nrounds: 60"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3265.3574759057597,
            "unit": "iter/sec",
            "range": "stddev: 0.000009960866920327357",
            "extra": "mean: 306.24518368317865 usec\nrounds: 2096"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 25.718914384047515,
            "unit": "iter/sec",
            "range": "stddev: 0.0007136341837602341",
            "extra": "mean: 38.88189000000182 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 19.518093565995,
            "unit": "iter/sec",
            "range": "stddev: 0.013780626716454329",
            "extra": "mean: 51.23451205000009 msec\nrounds: 20"
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
          "id": "29e2c3d168b4f2165e46bef284d6f4ccd318045a",
          "message": "docs: fix Returns rendering and tidy up docs styling (#36)\n\n* docs: fix Returns rendering in mkdocstrings\n\nIndent continuation lines in Returns docstring sections so griffe's\nGoogle-style parser treats them as a single entry instead of separate\nrows.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* docs: change theme to deep orange and add bottom padding\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* docs: tidy acknowledgements formatting and wording\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Add `/commit` and `/pr` skills\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-09T19:16:44+01:00",
          "tree_id": "f4c0e87ff247b203a80df1624b4cb61340461453",
          "url": "https://github.com/adrhill/asdex/commit/29e2c3d168b4f2165e46bef284d6f4ccd318045a"
        },
        "date": 1770661039241,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 677.145367624834,
            "unit": "iter/sec",
            "range": "stddev: 0.003731948593928422",
            "extra": "mean: 1.476787773809361 msec\nrounds: 168"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3181.6377425958362,
            "unit": "iter/sec",
            "range": "stddev: 0.000042526663793828266",
            "extra": "mean: 314.30353827275115 usec\nrounds: 2038"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 57.856557117330695,
            "unit": "iter/sec",
            "range": "stddev: 0.00030295826722973593",
            "extra": "mean: 17.284125600008338 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 78.70789344148659,
            "unit": "iter/sec",
            "range": "stddev: 0.0003088845332743432",
            "extra": "mean: 12.70520599999827 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 22.258785715505567,
            "unit": "iter/sec",
            "range": "stddev: 0.013007492560140353",
            "extra": "mean: 44.926080550000336 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 240.93621187356382,
            "unit": "iter/sec",
            "range": "stddev: 0.000039593595009196616",
            "extra": "mean: 4.150476145631319 msec\nrounds: 206"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 29.038709697686382,
            "unit": "iter/sec",
            "range": "stddev: 0.0004019125444288043",
            "extra": "mean: 34.436791800004585 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 11.348635814194116,
            "unit": "iter/sec",
            "range": "stddev: 0.0006530663978477299",
            "extra": "mean: 88.11631780000084 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 85.46581583166335,
            "unit": "iter/sec",
            "range": "stddev: 0.015358443790062229",
            "extra": "mean: 11.700584500002167 msec\nrounds: 58"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3246.857676684084,
            "unit": "iter/sec",
            "range": "stddev: 0.000008115087955776888",
            "extra": "mean: 307.9900936776722 usec\nrounds: 2167"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 25.90928707773267,
            "unit": "iter/sec",
            "range": "stddev: 0.0004120102207146526",
            "extra": "mean: 38.596198999988474 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 19.17292231189587,
            "unit": "iter/sec",
            "range": "stddev: 0.015491420064798632",
            "extra": "mean: 52.15688999999486 msec\nrounds: 20"
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
          "id": "5ef98b5c3697eb80c4b316c45366b441e33d4d9d",
          "message": "feat: validate input shape against colored pattern (#37)\n\n* feat: validate input shape against colored pattern in jacobian/hessian\n\n- Add `input_shape` field to `SparsityPattern`\n- Set it during sparsity detection\n- Raise `ValueError` when the input shape doesn't match the precomputed pattern\n- Tests for the shape mismatch error and `ColoredPattern.__repr__`\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* docs: add commit body to commit skill template\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-12T13:49:09+01:00",
          "tree_id": "d86c37176611f1dad5816d56b383798d10cae6fa",
          "url": "https://github.com/adrhill/asdex/commit/5ef98b5c3697eb80c4b316c45366b441e33d4d9d"
        },
        "date": 1770900583453,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 719.833868817538,
            "unit": "iter/sec",
            "range": "stddev: 0.00281169340054362",
            "extra": "mean: 1.3892094319522466 msec\nrounds: 169"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3250.358198832145,
            "unit": "iter/sec",
            "range": "stddev: 0.000016023514380197455",
            "extra": "mean: 307.6583991140732 usec\nrounds: 2032"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 60.896325710578616,
            "unit": "iter/sec",
            "range": "stddev: 0.0005793893723020273",
            "extra": "mean: 16.42135199999899 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 82.65035078213562,
            "unit": "iter/sec",
            "range": "stddev: 0.00031867566911456316",
            "extra": "mean: 12.099162200000535 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 20.227954461359545,
            "unit": "iter/sec",
            "range": "stddev: 0.01110437429137803",
            "extra": "mean: 49.43653605263203 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 242.89627924202574,
            "unit": "iter/sec",
            "range": "stddev: 0.00016368438762870383",
            "extra": "mean: 4.116983607655777 msec\nrounds: 209"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 23.847389261988344,
            "unit": "iter/sec",
            "range": "stddev: 0.01964484672127803",
            "extra": "mean: 41.93331140000112 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 11.548515658426918,
            "unit": "iter/sec",
            "range": "stddev: 0.0010206788107549611",
            "extra": "mean: 86.59121479999925 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 88.32167959160938,
            "unit": "iter/sec",
            "range": "stddev: 0.012334429811395198",
            "extra": "mean: 11.322248451613465 msec\nrounds: 62"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3219.0430446066243,
            "unit": "iter/sec",
            "range": "stddev: 0.000005888728460267756",
            "extra": "mean: 310.65132902632644 usec\nrounds: 2167"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 25.7300705538602,
            "unit": "iter/sec",
            "range": "stddev: 0.0013814747982198487",
            "extra": "mean: 38.865031400000305 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 19.65256932154206,
            "unit": "iter/sec",
            "range": "stddev: 0.012069248409218416",
            "extra": "mean: 50.883931949999806 msec\nrounds: 20"
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
          "id": "20c89579e4df0ab695892ab2b29f9b415391cc1c",
          "message": "build: replace pre-commit with prek",
          "timestamp": "2026-02-12T14:54:29+01:00",
          "tree_id": "2689ed4d19e5834e34d37251d6b920c08a14f37c",
          "url": "https://github.com/adrhill/asdex/commit/20c89579e4df0ab695892ab2b29f9b415391cc1c"
        },
        "date": 1770904504648,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 643.5313669814055,
            "unit": "iter/sec",
            "range": "stddev: 0.0044542216082207485",
            "extra": "mean: 1.5539258089169325 msec\nrounds: 157"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3224.23806200862,
            "unit": "iter/sec",
            "range": "stddev: 0.00002916052790188686",
            "extra": "mean: 310.15079555788907 usec\nrounds: 1981"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 60.15702813115722,
            "unit": "iter/sec",
            "range": "stddev: 0.0006391444736341843",
            "extra": "mean: 16.623161600000458 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 42.775986300461106,
            "unit": "iter/sec",
            "range": "stddev: 0.024326382584644674",
            "extra": "mean: 23.377602399999375 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 24.27935909236833,
            "unit": "iter/sec",
            "range": "stddev: 0.0018566868601466843",
            "extra": "mean: 41.187248649999475 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 241.69342933768402,
            "unit": "iter/sec",
            "range": "stddev: 0.0004251940551206898",
            "extra": "mean: 4.137472842105448 msec\nrounds: 209"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 30.199918264526453,
            "unit": "iter/sec",
            "range": "stddev: 0.0007513129798931189",
            "extra": "mean: 33.112672399998644 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 11.540333424929234,
            "unit": "iter/sec",
            "range": "stddev: 0.002217146288181997",
            "extra": "mean: 86.65260899999794 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 122.60313812162175,
            "unit": "iter/sec",
            "range": "stddev: 0.00020022473470073033",
            "extra": "mean: 8.156398076923647 msec\nrounds: 13"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3218.803335481656,
            "unit": "iter/sec",
            "range": "stddev: 0.000019914681849650234",
            "extra": "mean: 310.6744636979699 usec\nrounds: 2066"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 25.070076253194046,
            "unit": "iter/sec",
            "range": "stddev: 0.0014552440469540215",
            "extra": "mean: 39.88819140000004 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 18.838062094605796,
            "unit": "iter/sec",
            "range": "stddev: 0.01636441936288194",
            "extra": "mean: 53.08401655000097 msec\nrounds: 20"
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
          "id": "08f32fa961090daebddf2529fcadd9004e545be0",
          "message": "feat(interpret): add precise handler for `top_k` primitive (#38)\n\nValues output uses reduction-along-last-axis sparsity (each output\ndepends on all inputs in its batch slice). Indices output has zero\nderivative. More precise than conservative fallback for batched inputs.\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-12T15:17:31+01:00",
          "tree_id": "8651be7ee9313a8c141c10627ef89b2a0528945e",
          "url": "https://github.com/adrhill/asdex/commit/08f32fa961090daebddf2529fcadd9004e545be0"
        },
        "date": 1770905886118,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 722.9857522909932,
            "unit": "iter/sec",
            "range": "stddev: 0.003063088422791185",
            "extra": "mean: 1.3831531213875317 msec\nrounds: 173"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3519.8497612231654,
            "unit": "iter/sec",
            "range": "stddev: 0.000005846734298674359",
            "extra": "mean: 284.1030350262719 usec\nrounds: 1713"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 60.30748009959075,
            "unit": "iter/sec",
            "range": "stddev: 0.0014145538334847418",
            "extra": "mean: 16.581690999998955 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 88.84159673010656,
            "unit": "iter/sec",
            "range": "stddev: 0.0002949782751300568",
            "extra": "mean: 11.255988600001388 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 24.055162644675807,
            "unit": "iter/sec",
            "range": "stddev: 0.009461796350496083",
            "extra": "mean: 41.571117800000934 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 259.2180117685354,
            "unit": "iter/sec",
            "range": "stddev: 0.0002773595661799193",
            "extra": "mean: 3.8577566164381123 msec\nrounds: 219"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 31.4929704320252,
            "unit": "iter/sec",
            "range": "stddev: 0.00047238008533564204",
            "extra": "mean: 31.753117799999586 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 11.351340179178306,
            "unit": "iter/sec",
            "range": "stddev: 0.021916167253562754",
            "extra": "mean: 88.09532479999973 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 101.8270910945465,
            "unit": "iter/sec",
            "range": "stddev: 0.009808564689977936",
            "extra": "mean: 9.820569253731303 msec\nrounds: 67"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3485.344607331911,
            "unit": "iter/sec",
            "range": "stddev: 0.00000555427332546547",
            "extra": "mean: 286.9156748220419 usec\nrounds: 2248"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 30.104539397866976,
            "unit": "iter/sec",
            "range": "stddev: 0.00043993295837188205",
            "extra": "mean: 33.21758180000103 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 21.865084352220794,
            "unit": "iter/sec",
            "range": "stddev: 0.013924067876027369",
            "extra": "mean: 45.73501679166547 msec\nrounds: 24"
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
          "id": "be0797e8b6a3296ae249f790351b4fe6ac6d2c32",
          "message": "feat(interpret): dispatch `scatter-mul`, `scatter-min`, `scatter-max` primitives (#39)\n\n* feat(interpret): dispatch `scatter-mul`, `scatter-min`, `scatter-max` primitives\n\nRoute the three remaining scatter combine variants through prop_scatter.\nThe existing logic already handles them correctly via the update_jaxpr\ncheck \u2014 only the dispatch case and variable naming needed updating.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* refactor(test): parametrize scatter combine tests\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-12T15:45:45+01:00",
          "tree_id": "203092706018877ec1f3b0abc99bd1e1a3a26f04",
          "url": "https://github.com/adrhill/asdex/commit/be0797e8b6a3296ae249f790351b4fe6ac6d2c32"
        },
        "date": 1770907579354,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 722.2947220769785,
            "unit": "iter/sec",
            "range": "stddev: 0.0026879170408861657",
            "extra": "mean: 1.384476404762411 msec\nrounds: 168"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3231.9595786423333,
            "unit": "iter/sec",
            "range": "stddev: 0.000013487237915397988",
            "extra": "mean: 309.40981026132613 usec\nrounds: 2066"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 60.6559469977695,
            "unit": "iter/sec",
            "range": "stddev: 0.00046462734021231514",
            "extra": "mean: 16.486429599999042 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 76.44995870860156,
            "unit": "iter/sec",
            "range": "stddev: 0.0017894496803650183",
            "extra": "mean: 13.080451800001924 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 23.360653824333163,
            "unit": "iter/sec",
            "range": "stddev: 0.009192213273306632",
            "extra": "mean: 42.807021050000316 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 240.82145664731638,
            "unit": "iter/sec",
            "range": "stddev: 0.00004304719848695567",
            "extra": "mean: 4.152453913043565 msec\nrounds: 207"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 31.16700207867406,
            "unit": "iter/sec",
            "range": "stddev: 0.0008123277717946671",
            "extra": "mean: 32.08521619999658 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 10.560161708009309,
            "unit": "iter/sec",
            "range": "stddev: 0.023477852793254407",
            "extra": "mean: 94.69551960000331 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 98.22724488927103,
            "unit": "iter/sec",
            "range": "stddev: 0.009836096977747715",
            "extra": "mean: 10.180474888889265 msec\nrounds: 63"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3243.1624211003145,
            "unit": "iter/sec",
            "range": "stddev: 0.000006364695528610973",
            "extra": "mean: 308.3410172410446 usec\nrounds: 2146"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 25.80413596639022,
            "unit": "iter/sec",
            "range": "stddev: 0.0015930234754946624",
            "extra": "mean: 38.753477399998815 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 20.48316589060503,
            "unit": "iter/sec",
            "range": "stddev: 0.01135465483540263",
            "extra": "mean: 48.82057809523809 msec\nrounds: 21"
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
          "id": "a1102180c9bda9ca851452b9f86c895d234a15fc",
          "message": "feat(interpret): add precise handler for `platform_index` primitive (#40)\n\nEnables sparsity detection through `jax.lax.platform_dependent` and\nops that use it internally (e.g. `jnp.diag`). The output is a constant\nscalar with no input dependencies, so the handler is trivial.\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-12T23:35:04+01:00",
          "tree_id": "4165087bfaa2a36eeec963228ebf1a769e75fca8",
          "url": "https://github.com/adrhill/asdex/commit/a1102180c9bda9ca851452b9f86c895d234a15fc"
        },
        "date": 1770935740847,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 682.6385393583553,
            "unit": "iter/sec",
            "range": "stddev: 0.0041925798684314805",
            "extra": "mean: 1.4649041071427757 msec\nrounds: 168"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3495.11698356521,
            "unit": "iter/sec",
            "range": "stddev: 0.000007538006808316196",
            "extra": "mean: 286.11345620252905 usec\nrounds: 1975"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 60.814293908406455,
            "unit": "iter/sec",
            "range": "stddev: 0.00024124658769941485",
            "extra": "mean: 16.44350259999925 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 83.480956883445,
            "unit": "iter/sec",
            "range": "stddev: 0.00025827983433757726",
            "extra": "mean: 11.978779799999018 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 23.697651990430153,
            "unit": "iter/sec",
            "range": "stddev: 0.012758314959740056",
            "extra": "mean: 42.19827350000038 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 250.10697416341662,
            "unit": "iter/sec",
            "range": "stddev: 0.0006264984177472939",
            "extra": "mean: 3.998289145454269 msec\nrounds: 220"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 29.2947281312716,
            "unit": "iter/sec",
            "range": "stddev: 0.00034950355629731337",
            "extra": "mean: 34.1358348 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 10.160176771795076,
            "unit": "iter/sec",
            "range": "stddev: 0.028674849207392075",
            "extra": "mean: 98.42348439999853 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 88.44099091402681,
            "unit": "iter/sec",
            "range": "stddev: 0.013586433887862445",
            "extra": "mean: 11.306974171875762 msec\nrounds: 64"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3439.209209553748,
            "unit": "iter/sec",
            "range": "stddev: 0.00002398748706437811",
            "extra": "mean: 290.7645156398479 usec\nrounds: 2110"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 26.88348071995565,
            "unit": "iter/sec",
            "range": "stddev: 0.00024861250358308224",
            "extra": "mean: 37.19756419999953 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 18.95599433344662,
            "unit": "iter/sec",
            "range": "stddev: 0.020020676950717763",
            "extra": "mean: 52.75376128571452 msec\nrounds: 21"
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
          "id": "75f0b6e7343a883b7250de7fb14cb8a04c55cf65",
          "message": "feat(interpret): add precise handler for `scan` primitive (#41)\n\n* feat(interpret): add precise handler for `scan` primitive\n\nPropagate index sets through scan bodies using fixed-point iteration\non the carry, same strategy as while_loop. Also extract shared\n_MAX_FIXED_POINT_ITERS constant and raise RuntimeError on\nnon-convergence in both scan and while_loop.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* test(interpret): add edge-case and pytree test coverage for scan\n\nAdd 10 new tests for scan propagation: pytree xs/ys, length=1,\nscalar carry, unroll, ys-independent-of-carry, carry mixing,\ncarry tuple interaction, scan+cond composition, and pytree\nJacobian values.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* test(interpret): add tests for `associative_scan` decomposition\n\n`associative_scan` is not a JAX primitive \u2014 it decomposes into\nslice/add/pad/concatenate/rev, all of which have precise handlers.\nRemove the unreachable `prop_throw_error` dispatch entry and add\n8 tests verifying correct sparsity patterns through decomposition.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-13T00:35:17+01:00",
          "tree_id": "1ca5aeeb97d7b0389a38833c4af7bd42194232e0",
          "url": "https://github.com/adrhill/asdex/commit/75f0b6e7343a883b7250de7fb14cb8a04c55cf65"
        },
        "date": 1770939349410,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 677.5719615299944,
            "unit": "iter/sec",
            "range": "stddev: 0.003223352527064714",
            "extra": "mean: 1.475858000000392 msec\nrounds: 162"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3193.7044137508724,
            "unit": "iter/sec",
            "range": "stddev: 0.000040436034306314846",
            "extra": "mean: 313.11601527504604 usec\nrounds: 1964"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 60.32404096174191,
            "unit": "iter/sec",
            "range": "stddev: 0.0004958608321060904",
            "extra": "mean: 16.577138800005287 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 81.41699572296407,
            "unit": "iter/sec",
            "range": "stddev: 0.0004975323038944215",
            "extra": "mean: 12.282447799998408 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 22.980322295872117,
            "unit": "iter/sec",
            "range": "stddev: 0.010456052468856501",
            "extra": "mean: 43.51549064999958 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 241.7468017794468,
            "unit": "iter/sec",
            "range": "stddev: 0.000033911002445356324",
            "extra": "mean: 4.136559377990578 msec\nrounds: 209"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 30.22487811047109,
            "unit": "iter/sec",
            "range": "stddev: 0.0008562361882805677",
            "extra": "mean: 33.08532780000064 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 10.12057296409335,
            "unit": "iter/sec",
            "range": "stddev: 0.024852406334139383",
            "extra": "mean: 98.80863500000316 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 92.54643962133174,
            "unit": "iter/sec",
            "range": "stddev: 0.011824979326540029",
            "extra": "mean: 10.805385967214477 msec\nrounds: 61"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3241.553047861711,
            "unit": "iter/sec",
            "range": "stddev: 0.000011370589565621818",
            "extra": "mean: 308.494103053365 usec\nrounds: 2096"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 23.324174070120854,
            "unit": "iter/sec",
            "range": "stddev: 0.000377434627514299",
            "extra": "mean: 42.87397260000034 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 18.529680287023563,
            "unit": "iter/sec",
            "range": "stddev: 0.013524877257882071",
            "extra": "mean: 53.967471888886585 msec\nrounds: 18"
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
          "id": "14d521b2e0ec2a6d281547a44397ea1fe323f10c",
          "message": "test: add further Jacobian and Hessian sparsity tests (#42)\n\n* test: add SCT Global Jacobian and Hessian test cases\n\nPort 26 test cases from SparseConnectivityTracer.jl's Global Jacobian\nand Global Hessian testsets covering element-wise ops, conditional\nbranching, clamp interactions, composite functions, and various\nHessian sparsity patterns.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* test: add scalar and binary sparsity tests\n\nAdd test_scalar.py with parametrized Jacobian/Hessian tests for unary\n(R->R) and binary (R^2->R) functions covering nonlinear, linear,\nzero-derivative, and constant categories. Extend test_sct.py with\ncomposite function Jacobians and AMPGO07 benchmark. Add TODO for\nauto-squeezing (1,)-shaped output in hessian_sparsity.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* refactor(test): distribute test_sct.py into specific test files\n\nMove tests from test_sct.py to their natural homes: multi-variable\npatterns to test_detection.py, where/select Hessians to test_select.py,\nand scalar multiply-by-zero to test_scalar.py. Drop two duplicate\nJacobian where tests already in test_select.py. Remove SCT references\nfrom test docstrings.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-13T01:04:37+01:00",
          "tree_id": "c7b97f30a43100132fb6f21993c7f176266395d3",
          "url": "https://github.com/adrhill/asdex/commit/14d521b2e0ec2a6d281547a44397ea1fe323f10c"
        },
        "date": 1770941111644,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 727.4552136878331,
            "unit": "iter/sec",
            "range": "stddev: 0.0027085314256770636",
            "extra": "mean: 1.374655073170074 msec\nrounds: 164"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3250.6322520591775,
            "unit": "iter/sec",
            "range": "stddev: 0.000010122877984471044",
            "extra": "mean: 307.6324611516821 usec\nrounds: 2188"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 61.44268883429403,
            "unit": "iter/sec",
            "range": "stddev: 0.0003676199250515747",
            "extra": "mean: 16.275329400002647 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 82.83707985843277,
            "unit": "iter/sec",
            "range": "stddev: 0.0005073497606632555",
            "extra": "mean: 12.071888599996328 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 23.474296959974954,
            "unit": "iter/sec",
            "range": "stddev: 0.008539089758487146",
            "extra": "mean: 42.599784850002465 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 241.0727813514597,
            "unit": "iter/sec",
            "range": "stddev: 0.000028169016294428586",
            "extra": "mean: 4.148124870812775 msec\nrounds: 209"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 31.09607578531529,
            "unit": "iter/sec",
            "range": "stddev: 0.0005306354178229245",
            "extra": "mean: 32.15839860000074 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 11.6617692833323,
            "unit": "iter/sec",
            "range": "stddev: 0.0009188242088559065",
            "extra": "mean: 85.75028160000215 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 127.01147006716637,
            "unit": "iter/sec",
            "range": "stddev: 0.00018014594367225624",
            "extra": "mean: 7.873304666666551 msec\nrounds: 15"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3273.1664819843086,
            "unit": "iter/sec",
            "range": "stddev: 0.000010519743356701659",
            "extra": "mean: 305.51455463816336 usec\nrounds: 2059"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 24.152303070632144,
            "unit": "iter/sec",
            "range": "stddev: 0.0004760465592249132",
            "extra": "mean: 41.40391900000395 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 18.92967385755103,
            "unit": "iter/sec",
            "range": "stddev: 0.013274601161672946",
            "extra": "mean: 52.82711194736728 msec\nrounds: 19"
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
          "id": "d0cf6c855daafb2c2c89bfee477285c0970cdf63",
          "message": "feat: auto-squeeze scalar functions in Hessian API (#43)\n\nFunctions returning shape (1,) or (1, 1) are now automatically squeezed\nto scalar in hessian_sparsity, hessian, and _eval_hessian. Non-scalar\noutputs like (3,) raise ValueError with a clear message.\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-13T01:20:15+01:00",
          "tree_id": "f49c2009bdf44805368feebffef7c95d5eb0fba9",
          "url": "https://github.com/adrhill/asdex/commit/d0cf6c855daafb2c2c89bfee477285c0970cdf63"
        },
        "date": 1770942048460,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 721.1351423090557,
            "unit": "iter/sec",
            "range": "stddev: 0.0026441433518342708",
            "extra": "mean: 1.3867026321835132 msec\nrounds: 174"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3144.214328351857,
            "unit": "iter/sec",
            "range": "stddev: 0.000049835068120109474",
            "extra": "mean: 318.0444764794971 usec\nrounds: 1977"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 59.78561929148982,
            "unit": "iter/sec",
            "range": "stddev: 0.0005269357298390231",
            "extra": "mean: 16.72643039999997 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 80.96478020723963,
            "unit": "iter/sec",
            "range": "stddev: 0.00027498864592990667",
            "extra": "mean: 12.351049400002978 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 24.76357009312604,
            "unit": "iter/sec",
            "range": "stddev: 0.00039189592060926223",
            "extra": "mean: 40.38189955 msec\nrounds: 20"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 238.4630597217845,
            "unit": "iter/sec",
            "range": "stddev: 0.00023481792141646113",
            "extra": "mean: 4.1935216346158715 msec\nrounds: 208"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 30.180190707901723,
            "unit": "iter/sec",
            "range": "stddev: 0.0009931302215204461",
            "extra": "mean: 33.13431679999894 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 11.840995746276514,
            "unit": "iter/sec",
            "range": "stddev: 0.0016599872266401534",
            "extra": "mean: 84.45235699999785 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 92.97123639950034,
            "unit": "iter/sec",
            "range": "stddev: 0.009764910214135876",
            "extra": "mean: 10.756014857143217 msec\nrounds: 56"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3260.1031224121384,
            "unit": "iter/sec",
            "range": "stddev: 0.000009933553293226288",
            "extra": "mean: 306.73876330025524 usec\nrounds: 2218"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 24.746880737347322,
            "unit": "iter/sec",
            "range": "stddev: 0.00024044901602949465",
            "extra": "mean: 40.409133200000724 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 18.887063444266953,
            "unit": "iter/sec",
            "range": "stddev: 0.012850195064945185",
            "extra": "mean: 52.946293263156456 msec\nrounds: 19"
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
          "id": "009c32c2da6487adaf1bee356f91323a29c1944f",
          "message": "feat(interpret): add precise handlers for `sort`, `split`, and `tile` (#44)\n\n* feat(interpret): add precise handlers for `split` and `tile` primitives\n\n- Add `_split.py`: partitions input along an axis, each output element\n  maps to exactly one input element (selection matrix Jacobian)\n- Add `_tile.py`: repeats input via modular indexing, each output element\n  depends on exactly one input element\n- Convert `prop_reshape` size-mismatch branch from conservative fallback\n  to ValueError (should never occur in valid JAX code)\n- Remove `split` and `tile` from conservative fallback group\n- Delete TODO.md (all items resolved)\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* chore(test): suppress pytest progress output for coding agents\n\nAdd --tb=short -q --no-header defaults and a conftest hook\nto only show output on failures, reducing noise for AI agents.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* feat(interpret): add precise handler for `sort` primitive\n\nSort along one dimension only mixes elements within slices along that\ndimension, producing block-diagonal patterns for multi-dimensional arrays.\nThe conservative fallback was correct for 1D but overly dense for nD.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* docs: add TODO for conservative fallback primitives\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* refactor(interpret): simplify sort handler with moveaxis grouping\n\nReplace np.indices/ravel_multi_index batch mapping and 1D special case\nwith a single moveaxis+reshape to group flat indices by batch coordinates.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* refactor(interpret): use union_all to shorten sort handler\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-13T16:16:03+01:00",
          "tree_id": "c1e02f55448ce6f0caa17a9c359348d8ed74ceb9",
          "url": "https://github.com/adrhill/asdex/commit/009c32c2da6487adaf1bee356f91323a29c1944f"
        },
        "date": 1770995798173,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 690.9142078096987,
            "unit": "iter/sec",
            "range": "stddev: 0.0033936562156871588",
            "extra": "mean: 1.4473577018630857 msec\nrounds: 161"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3191.0828839391606,
            "unit": "iter/sec",
            "range": "stddev: 0.000009173216417859197",
            "extra": "mean: 313.3732455001521 usec\nrounds: 2000"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 59.27249555391687,
            "unit": "iter/sec",
            "range": "stddev: 0.00047878416547247963",
            "extra": "mean: 16.871231599998282 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 80.47854215120998,
            "unit": "iter/sec",
            "range": "stddev: 0.0005675141620069814",
            "extra": "mean: 12.42567239999346 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 21.30256428619717,
            "unit": "iter/sec",
            "range": "stddev: 0.016506883476296964",
            "extra": "mean: 46.9427054210531 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 238.9763874247778,
            "unit": "iter/sec",
            "range": "stddev: 0.00004580951208033439",
            "extra": "mean: 4.184513837438304 msec\nrounds: 203"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 27.95862863306994,
            "unit": "iter/sec",
            "range": "stddev: 0.00047804367689638296",
            "extra": "mean: 35.76713339999742 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 11.139021488914315,
            "unit": "iter/sec",
            "range": "stddev: 0.0007201386855225613",
            "extra": "mean: 89.77449240000226 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 79.730616940342,
            "unit": "iter/sec",
            "range": "stddev: 0.01620858435274737",
            "extra": "mean: 12.542233314816121 msec\nrounds: 54"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3112.5766852491256,
            "unit": "iter/sec",
            "range": "stddev: 0.00000803711231821955",
            "extra": "mean: 321.2772249882613 usec\nrounds: 2089"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 23.024054251745774,
            "unit": "iter/sec",
            "range": "stddev: 0.0004455812829041531",
            "extra": "mean: 43.432837199998175 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 17.100853164129088,
            "unit": "iter/sec",
            "range": "stddev: 0.01869145448156639",
            "extra": "mean: 58.47661461111247 msec\nrounds: 18"
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
          "id": "306e7e9c88dc4c2f230f56cbf95c11717b10b281",
          "message": "refactor(interpret): extract common patterns into _commons.py (#45)\n\n* refactor(interpret): extract common patterns into _commons.py\n\nAdd `permute_indices`, `position_map`, `fixed_point_loop`, and\n`conservative_indices` utilities to reduce duplication across handlers.\n\n- permute_indices: replaces repeated [in_indices[j].copy() for j in map]\n  pattern across 8+ handler files\n- position_map: replaces repeated np.arange(n).reshape(shape) pattern\n- fixed_point_loop: extracts ~22-line fixed-point iteration from\n  _while.py and _scan.py\n- Rename conservative_deps \u2192 conservative_indices for consistency\n- Simplify _transpose.py and _slice.py using numpy iota approach\n- Update CLAUDE.md with new naming conventions\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* refactor(interpret): standardize variable names across handlers\n\nUse `permutation_map` consistently for the flat map passed to\n`permute_indices` (was `perm`, `chunk_indices` in some files).\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* docs(interpret): reorganize and improve CLAUDE.md\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Remove skills\n\n* refactor(interpret): rename *_deps variables to *_indices\n\nRename list[IndexSets] variables from *_deps to *_indices across\nhandlers, and standardize dim/const_inputs naming for consistency.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* style(interpret): reorganize _commons.py by concept\n\nReorder functions into logical groups and replace banner comments\nwith lightweight section headers.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-17T17:10:38+01:00",
          "tree_id": "2d1e15584dc444162aa78f05a5a3112aeac9c8b3",
          "url": "https://github.com/adrhill/asdex/commit/306e7e9c88dc4c2f230f56cbf95c11717b10b281"
        },
        "date": 1771344672881,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 887.7797741654978,
            "unit": "iter/sec",
            "range": "stddev: 0.0028226101101665956",
            "extra": "mean: 1.1264054770114444 msec\nrounds: 174"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3238.7044076284596,
            "unit": "iter/sec",
            "range": "stddev: 0.00000939234181852142",
            "extra": "mean: 308.7654426395306 usec\nrounds: 1970"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 60.66982201512308,
            "unit": "iter/sec",
            "range": "stddev: 0.0005524263511376948",
            "extra": "mean: 16.482659200001137 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 84.75666471742497,
            "unit": "iter/sec",
            "range": "stddev: 0.0003307333416506016",
            "extra": "mean: 11.798482200001104 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 21.97196157982859,
            "unit": "iter/sec",
            "range": "stddev: 0.010746329791815787",
            "extra": "mean: 45.51254999999873 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 241.70633722754758,
            "unit": "iter/sec",
            "range": "stddev: 0.00035484520475322166",
            "extra": "mean: 4.137251887850083 msec\nrounds: 214"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 30.921926285850002,
            "unit": "iter/sec",
            "range": "stddev: 0.0007166286924814365",
            "extra": "mean: 32.339511800000764 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 10.173761870577348,
            "unit": "iter/sec",
            "range": "stddev: 0.027639930393593512",
            "extra": "mean: 98.29205880000131 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 91.49734793493127,
            "unit": "iter/sec",
            "range": "stddev: 0.011482099347955155",
            "extra": "mean: 10.929278526314821 msec\nrounds: 57"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3322.205091691788,
            "unit": "iter/sec",
            "range": "stddev: 0.000010898572352751898",
            "extra": "mean: 301.0048965672868 usec\nrounds: 2214"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 24.745150927852492,
            "unit": "iter/sec",
            "range": "stddev: 0.0003438746288250086",
            "extra": "mean: 40.41195799999855 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 19.225298006730156,
            "unit": "iter/sec",
            "range": "stddev: 0.012048987092800638",
            "extra": "mean: 52.01479840000047 msec\nrounds: 20"
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
          "id": "c058eb137c4fd7347790f11b05049ef861d4d470",
          "message": "feat(interpret): validate that auxiliary inputs have no index sets (#46)\n\n* feat(interpret): validate that auxiliary inputs have no index sets\n\nAdd `check_no_index_sets` to `_commons.py` and call it in handlers\nthat silently ignore auxiliary input dependencies (gather, scatter,\ndynamic_slice, dynamic_update_slice, conv).\nThis turns an incorrect-result scenario into an explicit error\nwith a link to the issue tracker.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* docs(interpret): improve error messages with contribution prompt\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-17T17:38:19+01:00",
          "tree_id": "742ef1de5e5e12919e3cb83196fca2e264ff1fc0",
          "url": "https://github.com/adrhill/asdex/commit/c058eb137c4fd7347790f11b05049ef861d4d470"
        },
        "date": 1771346335230,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 907.2011507143188,
            "unit": "iter/sec",
            "range": "stddev: 0.002562102892464047",
            "extra": "mean: 1.1022913707865256 msec\nrounds: 178"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3309.835067150202,
            "unit": "iter/sec",
            "range": "stddev: 0.00002930650476216587",
            "extra": "mean: 302.1298583500141 usec\nrounds: 2012"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 60.05211226218607,
            "unit": "iter/sec",
            "range": "stddev: 0.0004533062844772982",
            "extra": "mean: 16.652203600000348 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 83.89537783282647,
            "unit": "iter/sec",
            "range": "stddev: 0.0002874840349272712",
            "extra": "mean: 11.919607799998744 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 21.977162313798104,
            "unit": "iter/sec",
            "range": "stddev: 0.012082352870536843",
            "extra": "mean: 45.50177978947545 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 249.6088576956201,
            "unit": "iter/sec",
            "range": "stddev: 0.000033380100138361045",
            "extra": "mean: 4.006268083720921 msec\nrounds: 215"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 30.67801813057131,
            "unit": "iter/sec",
            "range": "stddev: 0.0008903631565513651",
            "extra": "mean: 32.596629800002574 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 11.551705826883836,
            "unit": "iter/sec",
            "range": "stddev: 0.0018878810816523204",
            "extra": "mean: 86.56730140000093 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 125.52423813601064,
            "unit": "iter/sec",
            "range": "stddev: 0.00012179748303217716",
            "extra": "mean: 7.966588882351623 msec\nrounds: 17"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3294.259441933651,
            "unit": "iter/sec",
            "range": "stddev: 0.000009281205300355939",
            "extra": "mean: 303.55836194037715 usec\nrounds: 2144"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 24.086868037055886,
            "unit": "iter/sec",
            "range": "stddev: 0.0004054246260009299",
            "extra": "mean: 41.51639800000453 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 19.848864788837783,
            "unit": "iter/sec",
            "range": "stddev: 0.0001705477214420844",
            "extra": "mean: 50.38071500000143 msec\nrounds: 10"
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
          "id": "e2c74a45c0a476d3264e2fcc91a299666504b897",
          "message": "style: replace decorative block comments with plain comments (#47)\n\n* style: replace decorative block comments with plain comments\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-17T17:46:42+01:00",
          "tree_id": "b8dd20dd152e00867cf74ff9d11418ba1624477d",
          "url": "https://github.com/adrhill/asdex/commit/e2c74a45c0a476d3264e2fcc91a299666504b897"
        },
        "date": 1771346835811,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 895.3563660514942,
            "unit": "iter/sec",
            "range": "stddev: 0.0029843897186507956",
            "extra": "mean: 1.1168737252743086 msec\nrounds: 182"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3349.833643737129,
            "unit": "iter/sec",
            "range": "stddev: 0.000008166385634820083",
            "extra": "mean: 298.5222868812028 usec\nrounds: 2081"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 59.93402941515513,
            "unit": "iter/sec",
            "range": "stddev: 0.0006845230057252801",
            "extra": "mean: 16.68501199999639 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 84.56250715929335,
            "unit": "iter/sec",
            "range": "stddev: 0.0002850108147459759",
            "extra": "mean: 11.825571799997192 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 22.47406642659974,
            "unit": "iter/sec",
            "range": "stddev: 0.010802625071786",
            "extra": "mean: 44.495730368422564 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 249.38295598678462,
            "unit": "iter/sec",
            "range": "stddev: 0.000032724505154932973",
            "extra": "mean: 4.009897132075828 msec\nrounds: 212"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 20.555906282712566,
            "unit": "iter/sec",
            "range": "stddev: 0.03016463244369432",
            "extra": "mean: 48.64781859999994 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 11.416835730404872,
            "unit": "iter/sec",
            "range": "stddev: 0.0014671698933717938",
            "extra": "mean: 87.58994380000047 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 82.3544950076212,
            "unit": "iter/sec",
            "range": "stddev: 0.016120217489593452",
            "extra": "mean: 12.142628036362298 msec\nrounds: 55"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3363.0489590851657,
            "unit": "iter/sec",
            "range": "stddev: 0.000014709834907808221",
            "extra": "mean: 297.3492245179878 usec\nrounds: 1452"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 24.357447609176244,
            "unit": "iter/sec",
            "range": "stddev: 0.0007014304993980885",
            "extra": "mean: 41.055204799999956 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 19.052117964141,
            "unit": "iter/sec",
            "range": "stddev: 0.012456449977708485",
            "extra": "mean: 52.48760278947217 msec\nrounds: 19"
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
          "id": "cc94199dc750725dee5cc07675d086bf16f39be7",
          "message": "docs: shorten acknowledgements",
          "timestamp": "2026-02-17T17:50:08+01:00",
          "tree_id": "8bd7686b3c7e4f8e423f153e559618cb66c5f1e8",
          "url": "https://github.com/adrhill/asdex/commit/cc94199dc750725dee5cc07675d086bf16f39be7"
        },
        "date": 1771347046708,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 871.0384654514165,
            "unit": "iter/sec",
            "range": "stddev: 0.003301625432639591",
            "extra": "mean: 1.1480549248554126 msec\nrounds: 173"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3279.9187003593042,
            "unit": "iter/sec",
            "range": "stddev: 0.00003263873323820441",
            "extra": "mean: 304.8856058201849 usec\nrounds: 1890"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 60.44568099653483,
            "unit": "iter/sec",
            "range": "stddev: 0.0004458521104396105",
            "extra": "mean: 16.5437791999949 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 84.96408296329852,
            "unit": "iter/sec",
            "range": "stddev: 0.0004478092277082204",
            "extra": "mean: 11.76967919999754 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 22.510432531042596,
            "unit": "iter/sec",
            "range": "stddev: 0.010672449601116109",
            "extra": "mean: 44.42384652631478 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 245.49786616382855,
            "unit": "iter/sec",
            "range": "stddev: 0.00003737099667877332",
            "extra": "mean: 4.07335516037711 msec\nrounds: 212"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 30.916384005384813,
            "unit": "iter/sec",
            "range": "stddev: 0.0005326903285155367",
            "extra": "mean: 32.345309199996564 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 10.164103826848775,
            "unit": "iter/sec",
            "range": "stddev: 0.025810664752011138",
            "extra": "mean: 98.38545700000338 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 88.14650392203067,
            "unit": "iter/sec",
            "range": "stddev: 0.012938536998624727",
            "extra": "mean: 11.344749428571125 msec\nrounds: 56"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3378.3095616257697,
            "unit": "iter/sec",
            "range": "stddev: 0.00000987223414928293",
            "extra": "mean: 296.0060295714175 usec\nrounds: 2029"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 24.366165970386394,
            "unit": "iter/sec",
            "range": "stddev: 0.0006311067479173439",
            "extra": "mean: 41.04051499999457 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 18.668754281445153,
            "unit": "iter/sec",
            "range": "stddev: 0.015304957419392397",
            "extra": "mean: 53.56543800000081 msec\nrounds: 19"
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
          "id": "eb5fa4de7d9df88a443e8a5cb4e2926c0662e97f",
          "message": "docs: shorten acknowledgements",
          "timestamp": "2026-02-17T17:51:11+01:00",
          "tree_id": "5d7f6a5da9c8e7cbb71e28d0f5a84ea8c7b0b3b6",
          "url": "https://github.com/adrhill/asdex/commit/eb5fa4de7d9df88a443e8a5cb4e2926c0662e97f"
        },
        "date": 1771347114292,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 824.7115165517603,
            "unit": "iter/sec",
            "range": "stddev: 0.003519076207313741",
            "extra": "mean: 1.2125452111802035 msec\nrounds: 161"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3270.479449794303,
            "unit": "iter/sec",
            "range": "stddev: 0.0000365374570701765",
            "extra": "mean: 305.7655659823501 usec\nrounds: 2046"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 57.852647640304504,
            "unit": "iter/sec",
            "range": "stddev: 0.0005640149218572437",
            "extra": "mean: 17.285293599999818 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 78.7457143235573,
            "unit": "iter/sec",
            "range": "stddev: 0.0005713139425574561",
            "extra": "mean: 12.699103799999989 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 22.363431704423686,
            "unit": "iter/sec",
            "range": "stddev: 0.013817636663093245",
            "extra": "mean: 44.715856368420916 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 248.39634721134718,
            "unit": "iter/sec",
            "range": "stddev: 0.00003402390819301327",
            "extra": "mean: 4.0258240961536895 msec\nrounds: 208"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 28.51044814410414,
            "unit": "iter/sec",
            "range": "stddev: 0.000540736487535653",
            "extra": "mean: 35.07486079999751 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 11.547173143355772,
            "unit": "iter/sec",
            "range": "stddev: 0.001232261856978557",
            "extra": "mean: 86.60128219999876 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 124.27185348503144,
            "unit": "iter/sec",
            "range": "stddev: 0.0001571928306139954",
            "extra": "mean: 8.046874428572437 msec\nrounds: 14"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3369.0482797394984,
            "unit": "iter/sec",
            "range": "stddev: 0.000009526468166210634",
            "extra": "mean: 296.81972977761006 usec\nrounds: 2250"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 23.401066769454253,
            "unit": "iter/sec",
            "range": "stddev: 0.00038090215105612753",
            "extra": "mean: 42.73309459999979 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 18.14602363080204,
            "unit": "iter/sec",
            "range": "stddev: 0.014614534069306845",
            "extra": "mean: 55.108492105264645 msec\nrounds: 19"
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
          "id": "eef8ea44791c20d5d98e1a2dbde6b443e0c7bc87",
          "message": "feat(interpret): add precise handler for `select_if_vmap` (#48)\n\n* feat(interpret): add precise handler for `select_if_vmap`\n\nReplace conservative fallback with element-wise union of both branches,\nmatching the existing `select_n` handler. This Equinox primitive appears\nwhen vmapping `lax.cond`.\n\nEquinox-specific handlers live in `_interpret/_equinox/`.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Delete TODO.md\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-17T18:17:49+01:00",
          "tree_id": "25eaa57189f044bee91577f16ac81633e07cd84a",
          "url": "https://github.com/adrhill/asdex/commit/eef8ea44791c20d5d98e1a2dbde6b443e0c7bc87"
        },
        "date": 1771348707903,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 794.7581487868484,
            "unit": "iter/sec",
            "range": "stddev: 0.004325144315467476",
            "extra": "mean: 1.2582444125001313 msec\nrounds: 160"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3299.206909097776,
            "unit": "iter/sec",
            "range": "stddev: 0.000008122032421213656",
            "extra": "mean: 303.10314798457637 usec\nrounds: 1811"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 56.99341974493999,
            "unit": "iter/sec",
            "range": "stddev: 0.0012365262502901367",
            "extra": "mean: 17.545885199997713 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 38.870789569521726,
            "unit": "iter/sec",
            "range": "stddev: 0.029183120711244577",
            "extra": "mean: 25.72625899999963 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 21.49567861794613,
            "unit": "iter/sec",
            "range": "stddev: 0.015405448979915474",
            "extra": "mean: 46.52097836842092 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 247.74740153217584,
            "unit": "iter/sec",
            "range": "stddev: 0.00008691777184810932",
            "extra": "mean: 4.036369276995733 msec\nrounds: 213"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 29.127138760652695,
            "unit": "iter/sec",
            "range": "stddev: 0.0013482713480052933",
            "extra": "mean: 34.332242799999335 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 9.752658668866257,
            "unit": "iter/sec",
            "range": "stddev: 0.03038615825784973",
            "extra": "mean: 102.53614260000035 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 83.60090009188005,
            "unit": "iter/sec",
            "range": "stddev: 0.015485337813165832",
            "extra": "mean: 11.961593701754026 msec\nrounds: 57"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3300.055113460094,
            "unit": "iter/sec",
            "range": "stddev: 0.000010027350713909727",
            "extra": "mean: 303.0252421910324 usec\nrounds: 2209"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 23.482686081700628,
            "unit": "iter/sec",
            "range": "stddev: 0.0003865794368745746",
            "extra": "mean: 42.58456620000004 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 17.514549784099458,
            "unit": "iter/sec",
            "range": "stddev: 0.016385578812295583",
            "extra": "mean: 57.095387111111904 msec\nrounds: 18"
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
          "id": "13fabd8e79dba77f2a2dd96261d06e920ee3e2f2",
          "message": "test: add Flax tests for ResNet and ViT (#49)\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-17T18:50:31+01:00",
          "tree_id": "9cf50e525acbf3d08e1bc76d246602ef308b32e4",
          "url": "https://github.com/adrhill/asdex/commit/13fabd8e79dba77f2a2dd96261d06e920ee3e2f2"
        },
        "date": 1771350669579,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 848.1519242675578,
            "unit": "iter/sec",
            "range": "stddev: 0.0034803017740738353",
            "extra": "mean: 1.1790340520226659 msec\nrounds: 173"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3302.1729983386026,
            "unit": "iter/sec",
            "range": "stddev: 0.000009883536333398007",
            "extra": "mean: 302.8308936276575 usec\nrounds: 1946"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 59.58449349304892,
            "unit": "iter/sec",
            "range": "stddev: 0.00039309783387273363",
            "extra": "mean: 16.782890000007455 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 84.4372846881196,
            "unit": "iter/sec",
            "range": "stddev: 0.0003901038652539448",
            "extra": "mean: 11.84310939999591 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 22.58239515505839,
            "unit": "iter/sec",
            "range": "stddev: 0.010303401372151659",
            "extra": "mean: 44.28228242104793 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 244.4707645469658,
            "unit": "iter/sec",
            "range": "stddev: 0.000036800575128853796",
            "extra": "mean: 4.090468657277373 msec\nrounds: 213"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 30.148011239108893,
            "unit": "iter/sec",
            "range": "stddev: 0.00108083669409251",
            "extra": "mean: 33.169683799997074 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 9.988891393836056,
            "unit": "iter/sec",
            "range": "stddev: 0.02812034583355739",
            "extra": "mean: 100.11120960000426 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 95.90259443405586,
            "unit": "iter/sec",
            "range": "stddev: 0.01152334325275721",
            "extra": "mean: 10.427246581818137 msec\nrounds: 55"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3331.3538941601664,
            "unit": "iter/sec",
            "range": "stddev: 0.00000881146345170166",
            "extra": "mean: 300.17825537928917 usec\nrounds: 2138"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 24.153194904191977,
            "unit": "iter/sec",
            "range": "stddev: 0.0007530479283163494",
            "extra": "mean: 41.40239019999967 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 18.747680535731813,
            "unit": "iter/sec",
            "range": "stddev: 0.012172765228445763",
            "extra": "mean: 53.33993173683899 msec\nrounds: 19"
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
          "id": "325e56cf39767a21ab5ffe52c20160d1d2c7c978",
          "message": "docs: update README",
          "timestamp": "2026-02-18T16:09:31+01:00",
          "tree_id": "1573b6be4af37e8cf42236cc3aeb4348467d9a93",
          "url": "https://github.com/adrhill/asdex/commit/325e56cf39767a21ab5ffe52c20160d1d2c7c978"
        },
        "date": 1771427406553,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 887.6147949517167,
            "unit": "iter/sec",
            "range": "stddev: 0.003667573302644708",
            "extra": "mean: 1.126614839779002 msec\nrounds: 181"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3546.3273002372307,
            "unit": "iter/sec",
            "range": "stddev: 0.00000865034274968596",
            "extra": "mean: 281.98186894173733 usec\nrounds: 2022"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 66.5987075694935,
            "unit": "iter/sec",
            "range": "stddev: 0.000590092441710684",
            "extra": "mean: 15.015306400000838 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 94.04145621991354,
            "unit": "iter/sec",
            "range": "stddev: 0.00029505833654987646",
            "extra": "mean: 10.633608199999856 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 21.275851685500108,
            "unit": "iter/sec",
            "range": "stddev: 0.011863085479565313",
            "extra": "mean: 47.00164368421118 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 243.5534595176675,
            "unit": "iter/sec",
            "range": "stddev: 0.00029745719547915423",
            "extra": "mean: 4.105874751195885 msec\nrounds: 209"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 32.176384038682585,
            "unit": "iter/sec",
            "range": "stddev: 0.0006900718021585829",
            "extra": "mean: 31.07869420000071 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 11.683695699884238,
            "unit": "iter/sec",
            "range": "stddev: 0.0009647985070291153",
            "extra": "mean: 85.58935679999848 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 131.55487469048475,
            "unit": "iter/sec",
            "range": "stddev: 0.0002750710137421722",
            "extra": "mean: 7.60139069230803 msec\nrounds: 13"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3526.538252676647,
            "unit": "iter/sec",
            "range": "stddev: 0.000009835950106978108",
            "extra": "mean: 283.56420045663725 usec\nrounds: 2190"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 26.64695246958439,
            "unit": "iter/sec",
            "range": "stddev: 0.00018536219295699606",
            "extra": "mean: 37.527743600001884 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 20.130791477727158,
            "unit": "iter/sec",
            "range": "stddev: 0.014416998759423516",
            "extra": "mean: 49.675145714285826 msec\nrounds: 21"
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
          "id": "ca7ac725914be33aadd47b1a78c2de9933833b39",
          "message": "feat(verify): add correctness check utilities (#50)\n\nVerify sparse Jacobians/Hessians against JAX's dense references\nto catch bugs where sparsity detection misses nonzeros.\n\n- Add src/asdex/verify.py with VerificationError exception\n- Export new functions and exception from __init__.py\n- Add tests for success and failure cases\n- Add verification sections to Jacobian/Hessian how-to guides\n- Add autodoc entries to reference pages\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-18T17:08:37+01:00",
          "tree_id": "d06e33916ebfcd6bbbd93937747d54217e16dd25",
          "url": "https://github.com/adrhill/asdex/commit/ca7ac725914be33aadd47b1a78c2de9933833b39"
        },
        "date": 1771430955334,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 918.8605859459819,
            "unit": "iter/sec",
            "range": "stddev: 0.002437972477932767",
            "extra": "mean: 1.0883043796796266 msec\nrounds: 187"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3332.409597918421,
            "unit": "iter/sec",
            "range": "stddev: 0.000009595319594334697",
            "extra": "mean: 300.08315923248057 usec\nrounds: 2085"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 60.3353690808572,
            "unit": "iter/sec",
            "range": "stddev: 0.000434996176379028",
            "extra": "mean: 16.57402640000214 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 83.67022311971093,
            "unit": "iter/sec",
            "range": "stddev: 0.00023865480859403327",
            "extra": "mean: 11.951683199998797 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 22.954167201720836,
            "unit": "iter/sec",
            "range": "stddev: 0.009227671391487405",
            "extra": "mean: 43.56507431578836 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 248.5233278954595,
            "unit": "iter/sec",
            "range": "stddev: 0.000036277982511482205",
            "extra": "mean: 4.02376713875587 msec\nrounds: 209"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 30.909148017958167,
            "unit": "iter/sec",
            "range": "stddev: 0.0007962053039007109",
            "extra": "mean: 32.35288139999852 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 10.48646857763906,
            "unit": "iter/sec",
            "range": "stddev: 0.020863033238656636",
            "extra": "mean: 95.36098759999732 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 96.6529247149234,
            "unit": "iter/sec",
            "range": "stddev: 0.009223274362946123",
            "extra": "mean: 10.346298396551244 msec\nrounds: 58"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3305.634132780921,
            "unit": "iter/sec",
            "range": "stddev: 0.0000074436950505923285",
            "extra": "mean: 302.51381726831727 usec\nrounds: 2189"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 24.03953742333521,
            "unit": "iter/sec",
            "range": "stddev: 0.00028299862425205686",
            "extra": "mean: 41.598138200001245 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 18.749027042678414,
            "unit": "iter/sec",
            "range": "stddev: 0.012038228777319972",
            "extra": "mean: 53.33610099999856 msec\nrounds: 19"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "me@sirmarcel.com",
            "name": "Marcel",
            "username": "sirmarcel"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "7abb59c29b919f8c5b5999431de9b8e1627db4c6",
          "message": "fix(interpret): shape-aware broadcasting, multi-dim scatter, const propagation (#51)\n\n* fix(interpret): propagate const_vals through convert_element_type\n\nJAX inserts convert_element_type (e.g. int64 \u2192 int32) before every\ngather/scatter operation. Without const value propagation through\nthis primitive, the chain from closure-captured index arrays to\ngather/scatter handlers breaks, causing all gathers to fall back\nto conservative (fully dense) index sets.\n\nAlso handles stop_gradient and bitcast_convert_type, which share\nthe same handler but lack the new_dtype parameter.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* fix(interpret): handle multi-dimensional scatter patterns\n\nExtend the scatter handler beyond the 1D-only case to support two\nadditional patterns:\n\n1. Batched scatter along dim 0 with trailing window dims preserved.\n   This is the pattern produced by the backward of features[indices]\n   on multi-dimensional arrays (scatter-add with update_window_dims=(1,),\n   inserted_window_dims=(0,)).\n\n2. Full-window scatter along an arbitrary single dimension.\n   This is the pattern from x.at[:, idx, :].set(value) where all update\n   dims are window dims and the scatter targets one position along an\n   arbitrary dimension.\n\nWithout these patterns, any scatter on multi-dimensional arrays fell\nback to conservative (fully dense) index sets.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* fix(interpret): use shape-aware broadcasting in binary elementwise ops\n\nThe previous implementation used flat modular indexing (i % len) to\nhandle broadcasting, which only works correctly for scalar (len=1)\nbroadcast. For multi-dimensional broadcasting like (16,16) * (16,1),\nit incorrectly maps output element [p,d] to in2[d] instead of in2[p]\nvia (p*16 + d) % 16 = d.\n\nUse numpy coordinate mapping (np.indices + ravel_multi_index) when\ninput shapes differ, respecting numpy broadcasting rules where size-1\ndimensions always read index 0. The fast path with modular indexing\nis preserved for the common cases of same-shape or scalar inputs.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* style: fix lint and formatting\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* test(benchmarks): add SchNet-style GNN Hessian benchmark\n\nAdds a minimal message-passing GNN as a fourth benchmark group\nexercising gather (neighbor lookup), broadcast (cutoff weighting),\nand scatter-add (gradient of gather) \u2014 the patterns pathological\nfor sparsity detection in graph neural networks.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* test(interpret): strengthen broadcast and scatter test assertions\n\nAdd test_binary_broadcast_dependent_operands that catches the flat\nmodular indexing bug by using input-dependent operands on both sides.\nUpdate test_scatter_2d to assert the exact conservative pattern with\na TODO documenting the true sparse result.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>\nCo-authored-by: adrhill <adrian.hill@mailbox.org>",
          "timestamp": "2026-02-18T21:29:10+01:00",
          "tree_id": "e18f7f37836df89f9b2a856397ce8ef6a8b1bef2",
          "url": "https://github.com/adrhill/asdex/commit/7abb59c29b919f8c5b5999431de9b8e1627db4c6"
        },
        "date": 1771446593490,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 801.3234340829583,
            "unit": "iter/sec",
            "range": "stddev: 0.004473012612609976",
            "extra": "mean: 1.247935549450652 msec\nrounds: 182"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3404.027838734071,
            "unit": "iter/sec",
            "range": "stddev: 0.000013354211350070177",
            "extra": "mean: 293.7696303834846 usec\nrounds: 2113"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 58.331140471314924,
            "unit": "iter/sec",
            "range": "stddev: 0.00026886058424906796",
            "extra": "mean: 17.14350160000322 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 85.81231966963263,
            "unit": "iter/sec",
            "range": "stddev: 0.0003430220766544421",
            "extra": "mean: 11.653338400009261 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 21.22945221571928,
            "unit": "iter/sec",
            "range": "stddev: 0.01911509782516312",
            "extra": "mean: 47.10437131578709 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 264.3864074775911,
            "unit": "iter/sec",
            "range": "stddev: 0.000034292143476003244",
            "extra": "mean: 3.7823427064220696 msec\nrounds: 218"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 27.337477928258593,
            "unit": "iter/sec",
            "range": "stddev: 0.0014302339892949402",
            "extra": "mean: 36.57981920001134 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 9.942281555884037,
            "unit": "iter/sec",
            "range": "stddev: 0.036136918766337296",
            "extra": "mean: 100.58053520000954 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 80.4429006992926,
            "unit": "iter/sec",
            "range": "stddev: 0.017362104570603294",
            "extra": "mean: 12.431177783334135 msec\nrounds: 60"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3292.1814424296067,
            "unit": "iter/sec",
            "range": "stddev: 0.000014272415786561456",
            "extra": "mean: 303.74996563433854 usec\nrounds: 2066"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 25.580019393643322,
            "unit": "iter/sec",
            "range": "stddev: 0.00039915045528182756",
            "extra": "mean: 39.09301180000284 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 19.07715300592426,
            "unit": "iter/sec",
            "range": "stddev: 0.017346199813022725",
            "extra": "mean: 52.418723049999016 msec\nrounds: 20"
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
          "id": "e30a9ad566853c81c1b634b4014e63760c9a4b73",
          "message": "fix(interpret): harden handlers with precise patterns (#52)\n\n* skills: harden `add-handler` skill based on PR #51 post-mortem\n\nAdd requirements to catch the class of bugs found in PR #51:\nconst propagation, broadcasting/non-square shapes, precision\nverification against jax.jacobian, conservative audit, and\ncomposition tests for the const chain.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* feat(interpret): harden handlers with precise patterns and tests\n\nPost-mortem on PR #51 revealed handlers that silently fell back to\nconservative for untested shapes. This hardens all five at-risk\nhandlers:\n\n- `_gather`: generalize to arbitrary dim + multi-dim collapse\n- `_scatter`: add multi-index pattern (`mat.at[rows, cols].set`)\n- `_conv`: add `feature_group_count` support, explicit batch_group fallback\n- `_concatenate`: propagate `const_vals` for downstream index resolution\n- `_dot_general`, `_pad`: no handler changes, tests only\n\nAdds ~65 new tests covering precision verification against\n`jax.jacobian`, non-square/broadcasting shapes, conservative audits,\nconst chain composition, and edge cases.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* chore: remove redundant TODO.md\n\nTracked in .plan/ files and inline TODO comments instead.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* docs: emphasize documenting conservative patterns with TODO notes\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-18T22:37:16+01:00",
          "tree_id": "099562dcfa6a04d641eba4b5285a640270d2b198",
          "url": "https://github.com/adrhill/asdex/commit/e30a9ad566853c81c1b634b4014e63760c9a4b73"
        },
        "date": 1771450744523,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 872.3550057660034,
            "unit": "iter/sec",
            "range": "stddev: 0.0029335394988741154",
            "extra": "mean: 1.1463223038674641 msec\nrounds: 181"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3304.9614632293815,
            "unit": "iter/sec",
            "range": "stddev: 0.000006140969304036674",
            "extra": "mean: 302.57538888906396 usec\nrounds: 2034"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 57.01338983405386,
            "unit": "iter/sec",
            "range": "stddev: 0.0004069893390734167",
            "extra": "mean: 17.53973940000151 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 78.6683103938378,
            "unit": "iter/sec",
            "range": "stddev: 0.00041383847120407845",
            "extra": "mean: 12.711598799995727 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 19.993370177403936,
            "unit": "iter/sec",
            "range": "stddev: 0.018540773220447847",
            "extra": "mean: 50.01658005263054 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 246.7515199867785,
            "unit": "iter/sec",
            "range": "stddev: 0.00010669067915464349",
            "extra": "mean: 4.052659939252176 msec\nrounds: 214"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 27.641446019788606,
            "unit": "iter/sec",
            "range": "stddev: 0.0006179811480704676",
            "extra": "mean: 36.17755740000348 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 9.58663334332417,
            "unit": "iter/sec",
            "range": "stddev: 0.03005802880551137",
            "extra": "mean: 104.31190640000523 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 84.239925543096,
            "unit": "iter/sec",
            "range": "stddev: 0.015067889444766652",
            "extra": "mean: 11.870855696428809 msec\nrounds: 56"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3344.78416465913,
            "unit": "iter/sec",
            "range": "stddev: 0.00000984493894661348",
            "extra": "mean: 298.97295334209133 usec\nrounds: 2229"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 22.79437688616623,
            "unit": "iter/sec",
            "range": "stddev: 0.00020778867396821323",
            "extra": "mean: 43.87046880000014 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 17.357844053582088,
            "unit": "iter/sec",
            "range": "stddev: 0.017645249504226454",
            "extra": "mean: 57.61084135294053 msec\nrounds: 17"
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
          "id": "b0bfd969dcc656852f3640911d6ecc492ff8189c",
          "message": "test: mark conservative sparsity patterns with fallback TODOs (#53)\n\nAudit all test files for overly conservative sparsity patterns\nand document precise expected results as TODO comments.\n\nFiles edited:\n- test_detection.py: multiply-by-zero (elementwise)\n- test_scalar.py: multiply-by-zero Hessian\n- test_gather.py: dynamic indices fallback\n- test_scatter.py: dynamic indices fallback\n- test_dynamic_slice.py: dynamic start fallback (2 tests)\n- test_dot_general.py: constant matrix zeros ignored\n- test_internals.py: eye @ x gives dense pattern\n- test_platform_index.py: diag via dynamic_update_slice\n- test_scan.py: deps merged across time steps (9 tests)\n- TODO.md: consolidated roadmap of all findings\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-18T23:55:36+01:00",
          "tree_id": "b3eec09525f0fe6c930d09af901e44950e75813e",
          "url": "https://github.com/adrhill/asdex/commit/b0bfd969dcc656852f3640911d6ecc492ff8189c"
        },
        "date": 1771455395524,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 751.0773195559525,
            "unit": "iter/sec",
            "range": "stddev: 0.005351251542570341",
            "extra": "mean: 1.3314208457142789 msec\nrounds: 175"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3281.0376203929595,
            "unit": "iter/sec",
            "range": "stddev: 0.000008899686722998263",
            "extra": "mean: 304.7816318181177 usec\nrounds: 1980"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 57.307145923735796,
            "unit": "iter/sec",
            "range": "stddev: 0.00028495373566714434",
            "extra": "mean: 17.449830799998267 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 79.06357981577938,
            "unit": "iter/sec",
            "range": "stddev: 0.00021345568981382388",
            "extra": "mean: 12.64804860000055 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 20.826046774546683,
            "unit": "iter/sec",
            "range": "stddev: 0.01583780482649812",
            "extra": "mean: 48.016794105263735 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 249.81001918858271,
            "unit": "iter/sec",
            "range": "stddev: 0.00005499882378093921",
            "extra": "mean: 4.00304200467274 msec\nrounds: 214"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 27.8888020140331,
            "unit": "iter/sec",
            "range": "stddev: 0.00044473371330233923",
            "extra": "mean: 35.856685399997446 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 11.031338939619024,
            "unit": "iter/sec",
            "range": "stddev: 0.0007342707566974999",
            "extra": "mean: 90.6508271999968 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 115.06762780771484,
            "unit": "iter/sec",
            "range": "stddev: 0.0003437455676104372",
            "extra": "mean: 8.690541545455877 msec\nrounds: 11"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3303.1923826390694,
            "unit": "iter/sec",
            "range": "stddev: 0.00001482501621781785",
            "extra": "mean: 302.7374382599705 usec\nrounds: 2138"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 22.81542322610094,
            "unit": "iter/sec",
            "range": "stddev: 0.0008605542979479484",
            "extra": "mean: 43.829999999999814 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 18.528880489930103,
            "unit": "iter/sec",
            "range": "stddev: 0.00034191939237370405",
            "extra": "mean: 53.96980138888966 msec\nrounds: 18"
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
          "id": "e8dbf8d1a5c6256c132aba7d643536eca9dc9f38",
          "message": "feat(coloring): warn when coloring yields no speedup over dense AD (#54)\n\nEmit a DenseColoringWarning when color_jacobian_pattern or\ncolor_hessian_pattern produces as many colors as the dense baseline,\nsince sparse differentiation offers no performance benefit in that case.\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-19T00:13:50+01:00",
          "tree_id": "9ef80f0beeb4d6e7a0df654ed472657b97ce2b48",
          "url": "https://github.com/adrhill/asdex/commit/e8dbf8d1a5c6256c132aba7d643536eca9dc9f38"
        },
        "date": 1771456473444,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 884.4956460172849,
            "unit": "iter/sec",
            "range": "stddev: 0.0026486268575055547",
            "extra": "mean: 1.130587815217417 msec\nrounds: 184"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3297.9303940879768,
            "unit": "iter/sec",
            "range": "stddev: 0.000008832199711216914",
            "extra": "mean: 303.2204687499307 usec\nrounds: 1952"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 40.77742952312364,
            "unit": "iter/sec",
            "range": "stddev: 0.0170371099112675",
            "extra": "mean: 24.523370200000727 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 83.34650624865192,
            "unit": "iter/sec",
            "range": "stddev: 0.0004214410016286995",
            "extra": "mean: 11.998103399999138 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 21.922371688811516,
            "unit": "iter/sec",
            "range": "stddev: 0.012930276512139398",
            "extra": "mean: 45.61550247368392 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 248.94822186966275,
            "unit": "iter/sec",
            "range": "stddev: 0.00004339211380217952",
            "extra": "mean: 4.01689954838702 msec\nrounds: 217"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 30.40473321427657,
            "unit": "iter/sec",
            "range": "stddev: 0.0008342170834765767",
            "extra": "mean: 32.88961599999993 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 10.463897999060581,
            "unit": "iter/sec",
            "range": "stddev: 0.02287484335432458",
            "extra": "mean: 95.56668080000179 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 94.45369400194704,
            "unit": "iter/sec",
            "range": "stddev: 0.010573741783773957",
            "extra": "mean: 10.587198421052609 msec\nrounds: 57"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3269.446607711375,
            "unit": "iter/sec",
            "range": "stddev: 0.00000878377379941122",
            "extra": "mean: 305.86215955978065 usec\nrounds: 2181"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 24.587115091848887,
            "unit": "iter/sec",
            "range": "stddev: 0.0003198796690569107",
            "extra": "mean: 40.67170939999869 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 19.081877646435583,
            "unit": "iter/sec",
            "range": "stddev: 0.012468107953978656",
            "extra": "mean: 52.40574426315934 msec\nrounds: 19"
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
          "id": "3bb1ecfe1bc47293e69e6696a01ad0475c28adff",
          "message": "docs: update `CLAUDE.md` files\n\nReplace per-file tree listings in `_interpret`, `tests`, and `docs`\n`CLAUDE.md` with naming conventions that won't go stale. Add missing\nmodules (`verify.py`, `_display.py`, `_pad.py`, `_sort.py`) to root\n`CLAUDE.md` and split the architecture diagram into separate Jacobian\nand Hessian sections.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-19T11:32:13+01:00",
          "tree_id": "613a60c6674af76a4a831cc0f0f3d049ab7b9e4f",
          "url": "https://github.com/adrhill/asdex/commit/3bb1ecfe1bc47293e69e6696a01ad0475c28adff"
        },
        "date": 1771497182206,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 798.5158540722431,
            "unit": "iter/sec",
            "range": "stddev: 0.00421201749829527",
            "extra": "mean: 1.2523232881354014 msec\nrounds: 177"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3351.668446662795,
            "unit": "iter/sec",
            "range": "stddev: 0.00002488966085476916",
            "extra": "mean: 298.3588669087137 usec\nrounds: 1931"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 59.18611959546998,
            "unit": "iter/sec",
            "range": "stddev: 0.000590905820356176",
            "extra": "mean: 16.895853400001215 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 82.66840927274151,
            "unit": "iter/sec",
            "range": "stddev: 0.0002784418322299438",
            "extra": "mean: 12.096519199985778 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 21.626501495559452,
            "unit": "iter/sec",
            "range": "stddev: 0.012772713965640347",
            "extra": "mean: 46.23956399999921 msec\nrounds: 18"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 245.59552810014853,
            "unit": "iter/sec",
            "range": "stddev: 0.00003513028564607417",
            "extra": "mean: 4.07173537619228 msec\nrounds: 210"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 29.052812254406376,
            "unit": "iter/sec",
            "range": "stddev: 0.0008359552020113034",
            "extra": "mean: 34.420075800005634 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 9.699915837710867,
            "unit": "iter/sec",
            "range": "stddev: 0.03075994940635435",
            "extra": "mean: 103.09367799999336 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 87.54861437761215,
            "unit": "iter/sec",
            "range": "stddev: 0.01369740126866841",
            "extra": "mean: 11.422225321429178 msec\nrounds: 56"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3285.9717448275483,
            "unit": "iter/sec",
            "range": "stddev: 0.0000072548857494248426",
            "extra": "mean: 304.32398013589165 usec\nrounds: 2215"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 24.49107221312907,
            "unit": "iter/sec",
            "range": "stddev: 0.00041541869706479036",
            "extra": "mean: 40.83120539997935 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 18.71102102251398,
            "unit": "iter/sec",
            "range": "stddev: 0.013608529684565072",
            "extra": "mean: 53.444437842101344 msec\nrounds: 19"
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
          "id": "ed9d21b04d55f13a05bc74c61e55846e6dce753d",
          "message": "test(interpret): harden non-symmetric shape coverage and inline expected patterns (#55)\n\n* test(interpret): use non-square shapes to catch dimension transposition bugs\n\nReplace square and repeated-size dimensions in handler tests with\nasymmetric shapes so that axis-confusion bugs cannot hide behind\nmatching dimension sizes.\n\nAlso add \"Writing handler tests\" section to tests/CLAUDE.md\ndocumenting the non-square shape guideline and other test conventions.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* test: suppress `DenseColoringWarning` in small Jacobian-value tests\n\nThese tests intentionally use small inputs where coloring cannot\nbeat dense AD. The warning is expected, not a sign of conservatism.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* test(gather): add fallback tests for `const_vals` through `slice`, `transpose`, `tile`\n\nCover all four shape-transforming ops from TODO #1 that don't propagate\n`const_vals` (`reshape`, `slice`, `transpose`, `tile`). Convert the existing\n`reshape` xfail test to assert the conservative pattern directly, avoiding\nunnecessary Jacobian computation at test time.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* test(interpret): replace `_check_precision` helpers with explicit patterns\n\nInline all `_check_precision` and `_check_sparser_than_conservative`\ncalls in `test_gather.py` and `test_scatter.py` with pre-computed\nexpected sparsity patterns, avoiding runtime Jacobian computation.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-19T15:30:26+01:00",
          "tree_id": "e3c2d692eea9aae41aa3c1471e132d2af8eee0f6",
          "url": "https://github.com/adrhill/asdex/commit/ed9d21b04d55f13a05bc74c61e55846e6dce753d"
        },
        "date": 1771511469857,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 849.7821333284803,
            "unit": "iter/sec",
            "range": "stddev: 0.0031610397775889924",
            "extra": "mean: 1.176772211111496 msec\nrounds: 180"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3317.5394962346672,
            "unit": "iter/sec",
            "range": "stddev: 0.00001288833276385827",
            "extra": "mean: 301.42821242519574 usec\nrounds: 1996"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 57.90704617338086,
            "unit": "iter/sec",
            "range": "stddev: 0.0005250805392087143",
            "extra": "mean: 17.269055600002048 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 83.95340337609102,
            "unit": "iter/sec",
            "range": "stddev: 0.0004045805496032268",
            "extra": "mean: 11.911369400002059 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 23.692004637998284,
            "unit": "iter/sec",
            "range": "stddev: 0.0007522056222313504",
            "extra": "mean: 42.20833210526035 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 247.83505731755147,
            "unit": "iter/sec",
            "range": "stddev: 0.00015950712656648698",
            "extra": "mean: 4.034941669768286 msec\nrounds: 215"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 29.890910015248863,
            "unit": "iter/sec",
            "range": "stddev: 0.000917704085936548",
            "extra": "mean: 33.45498679999537 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 10.18143506663311,
            "unit": "iter/sec",
            "range": "stddev: 0.025774078515908308",
            "extra": "mean: 98.21798140001192 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 93.33815689646414,
            "unit": "iter/sec",
            "range": "stddev: 0.011331668815909409",
            "extra": "mean: 10.71373201754193 msec\nrounds: 57"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3233.095414329105,
            "unit": "iter/sec",
            "range": "stddev: 0.00004977156727196325",
            "extra": "mean: 309.3011098800215 usec\nrounds: 2166"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 22.682100293533715,
            "unit": "iter/sec",
            "range": "stddev: 0.0011372360346340676",
            "extra": "mean: 44.08762799999977 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 17.29457246209902,
            "unit": "iter/sec",
            "range": "stddev: 0.019756210719187612",
            "extra": "mean: 57.82160861111171 msec\nrounds: 18"
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
          "id": "c76f7134924f89cb8ca5e3228fe62d1340baa076",
          "message": "docs: add explanation pages and improve cross-references (#56)\n\n* docs(explanation): add ASD overview and global sparsity pages\n\nAdd two new explanation pages that provide the conceptual foundation\nfor the existing sparsity detection and graph coloring pages:\n- `asd.md`: AD basics, structural orthogonality, seed matrices,\n  compression/decompression, amortization, and Hessian extension\n- `global-sparsity.md`: local vs. global patterns, conservatism,\n  and the precision-over-speed design decision\n\nMove overlapping content out of `sparsity-detection.md` and\n`coloring.md` into the new pages and update nav + links.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* docs(explanation): expand coloring and sparsity detection pages\n\nColoring: add \"Row and Column Coloring\" and \"From Coloring to\nDecompression\" sections, attribute SparseMatrixColorings.jl in\nreferences, and link to API docs.\n\nSparsity detection: add sections on abstract interpretation motivation,\njaxpr representation, index set propagation with worked example,\nprimitive handler families, and fallback handlers.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* docs(explanation): reduce bold formatting to first-use definitions\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* docs(explanation): revise ASD overview and global sparsity pages\n\nUse `c` for number of colors, restructure Hessian section to separate\nthe definition-based argument from the symmetry-based one, and add a\ntip admonition encouraging users to report conservative patterns.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* docs: add cross-references and contribution tips across pages\n\nAdd links between explanation, how-to, tutorial, and reference pages\nwhere they help the reader navigate. Add tip admonitions encouraging\nusers to report incorrect or overly dense patterns. Move the \"verify\ncorrectness\" admonitions above the first code block in the how-to\nguides.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-19T18:33:02+01:00",
          "tree_id": "952f89e921ff1b66bec803f366c1b01ac4037b9c",
          "url": "https://github.com/adrhill/asdex/commit/c76f7134924f89cb8ca5e3228fe62d1340baa076"
        },
        "date": 1771522425826,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 882.1548986779286,
            "unit": "iter/sec",
            "range": "stddev: 0.0028373623623867356",
            "extra": "mean: 1.1335877650270763 msec\nrounds: 183"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3291.1961086860033,
            "unit": "iter/sec",
            "range": "stddev: 0.000017116509600791605",
            "extra": "mean: 303.8409037251949 usec\nrounds: 2067"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 59.42403656641067,
            "unit": "iter/sec",
            "range": "stddev: 0.00043975286622609743",
            "extra": "mean: 16.82820720000109 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 84.12014409108957,
            "unit": "iter/sec",
            "range": "stddev: 0.0003672919011332168",
            "extra": "mean: 11.887758999998255 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 21.396285294739663,
            "unit": "iter/sec",
            "range": "stddev: 0.01403423495561684",
            "extra": "mean: 46.73708478947291 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 249.1319145331875,
            "unit": "iter/sec",
            "range": "stddev: 0.00004941991824181203",
            "extra": "mean: 4.013937764150997 msec\nrounds: 212"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 31.114181154280576,
            "unit": "iter/sec",
            "range": "stddev: 0.0010870628057511403",
            "extra": "mean: 32.13968559999927 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 10.372656694455081,
            "unit": "iter/sec",
            "range": "stddev: 0.025501523876671754",
            "extra": "mean: 96.40731680000272 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 93.73807499938631,
            "unit": "iter/sec",
            "range": "stddev: 0.012160783683358165",
            "extra": "mean: 10.668023639343424 msec\nrounds: 61"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3166.15074011007,
            "unit": "iter/sec",
            "range": "stddev: 0.00006986948186713662",
            "extra": "mean: 315.8409318076989 usec\nrounds: 2185"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 25.189375107475563,
            "unit": "iter/sec",
            "range": "stddev: 0.001255299686810179",
            "extra": "mean: 39.699277799996935 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 20.324277820702118,
            "unit": "iter/sec",
            "range": "stddev: 0.011281365271957652",
            "extra": "mean: 49.2022402380964 msec\nrounds: 21"
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
          "id": "c0c090b7879293c25257b17b84ca2323651b45fe",
          "message": "feat(pattern): add save/load for `SparsityPattern` and `ColoredPattern` (#57)\n\nSerialize patterns to `.npz` files for reuse across sessions.\nAdd roundtrip tests and document `save`/`load` in the how-to guides.\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-19T18:55:07+01:00",
          "tree_id": "f62143cd5178fc75e8f41d5d63b8c39d9690ddcd",
          "url": "https://github.com/adrhill/asdex/commit/c0c090b7879293c25257b17b84ca2323651b45fe"
        },
        "date": 1771523749365,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 818.0080670852859,
            "unit": "iter/sec",
            "range": "stddev: 0.00394124037437695",
            "extra": "mean: 1.222481831460642 msec\nrounds: 178"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3058.3166302886725,
            "unit": "iter/sec",
            "range": "stddev: 0.00005998960565478144",
            "extra": "mean: 326.9772626209768 usec\nrounds: 1961"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 58.152149843107516,
            "unit": "iter/sec",
            "range": "stddev: 0.0005281605586323503",
            "extra": "mean: 17.19626880000078 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 82.15132365827267,
            "unit": "iter/sec",
            "range": "stddev: 0.00042472096397308556",
            "extra": "mean: 12.172658400000103 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 20.408668286680765,
            "unit": "iter/sec",
            "range": "stddev: 0.017191695028135303",
            "extra": "mean: 48.99878747368472 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 242.91939108328776,
            "unit": "iter/sec",
            "range": "stddev: 0.00005880460522324059",
            "extra": "mean: 4.116591909524169 msec\nrounds: 210"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 29.460122283545928,
            "unit": "iter/sec",
            "range": "stddev: 0.0010227344767580174",
            "extra": "mean: 33.94419040000116 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 11.215141215832125,
            "unit": "iter/sec",
            "range": "stddev: 0.0006967969009466868",
            "extra": "mean: 89.16517239999848 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 124.39467160592126,
            "unit": "iter/sec",
            "range": "stddev: 0.00015238817878864767",
            "extra": "mean: 8.03892953846103 msec\nrounds: 13"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3291.6772976725583,
            "unit": "iter/sec",
            "range": "stddev: 0.000008502756592987899",
            "extra": "mean: 303.7964871912166 usec\nrounds: 2186"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 23.271863096210154,
            "unit": "iter/sec",
            "range": "stddev: 0.00020535079843341978",
            "extra": "mean: 42.97034559999844 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 17.80996477284918,
            "unit": "iter/sec",
            "range": "stddev: 0.01666117689111819",
            "extra": "mean: 56.148342388889695 msec\nrounds: 18"
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
          "id": "64b97dc1add960dd9c0d38f7bf66aca326a29c38",
          "message": "feat(decompression): add `hvp_mode` parameter to `hessian()` (#58)\n\nSupport three AD composition strategies for Hessian-vector products:\n`fwd_over_rev` (default), `rev_over_fwd`, and `rev_over_rev`.\nAll produce the same result but differ in memory/performance\ncharacteristics.\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-19T19:28:48+01:00",
          "tree_id": "71bf45f0baa40e41b025b3e5b1d7c78bdedfa7e9",
          "url": "https://github.com/adrhill/asdex/commit/64b97dc1add960dd9c0d38f7bf66aca326a29c38"
        },
        "date": 1771525773744,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 771.410192440663,
            "unit": "iter/sec",
            "range": "stddev: 0.004504040021748754",
            "extra": "mean: 1.2963271807909384 msec\nrounds: 177"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3352.7207515154037,
            "unit": "iter/sec",
            "range": "stddev: 0.000021807882723399147",
            "extra": "mean: 298.26522222228255 usec\nrounds: 1998"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 55.39888013601172,
            "unit": "iter/sec",
            "range": "stddev: 0.00020892651752188528",
            "extra": "mean: 18.0509064000006 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 80.99214207756651,
            "unit": "iter/sec",
            "range": "stddev: 0.00026537025967727866",
            "extra": "mean: 12.346876800002349 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 22.719319932600254,
            "unit": "iter/sec",
            "range": "stddev: 0.0005439059878447428",
            "extra": "mean: 44.015401999999426 msec\nrounds: 9"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 247.6778234152053,
            "unit": "iter/sec",
            "range": "stddev: 0.00003814032216731452",
            "extra": "mean: 4.037503181395483 msec\nrounds: 215"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 28.887462862495596,
            "unit": "iter/sec",
            "range": "stddev: 0.000555566179971964",
            "extra": "mean: 34.617093399998566 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 9.716783190751729,
            "unit": "iter/sec",
            "range": "stddev: 0.03109141252410166",
            "extra": "mean: 102.91471780000023 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 82.66905669941337,
            "unit": "iter/sec",
            "range": "stddev: 0.015624223047501802",
            "extra": "mean: 12.096424465517048 msec\nrounds: 58"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3283.4065366828754,
            "unit": "iter/sec",
            "range": "stddev: 0.000027074195435156823",
            "extra": "mean: 304.5617375819289 usec\nrounds: 2134"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 23.78740062076028,
            "unit": "iter/sec",
            "range": "stddev: 0.0007957688054684066",
            "extra": "mean: 42.03906159999917 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 18.230036331349755,
            "unit": "iter/sec",
            "range": "stddev: 0.014206668939813377",
            "extra": "mean: 54.85452589473582 msec\nrounds: 19"
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
          "id": "18f12d46d7e0b4f42958d5d784c84b2f55d6557c",
          "message": "feat(interpret): improve sparsity via const-value tracking (#59)\n\n* feat(interpret): improve sparsity for const_vals, mul-by-zero, and dot_general\n\nPropagate const_vals through reshape, slice, transpose, and tile so that\nconstant index arrays survive shape-transforming ops before gather/scatter.\n\nAdd prop_mul handler in _mul.py that clears dependencies at positions\nwhere a known-zero constant makes the product zero (d(0*y)/dy = 0).\n\nMake dot_general skip contraction terms where either factor is a known\nconstant zero, producing sparser patterns for matmul with sparse constants.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* refactor(interpret): extract `propagate_const_unary` and `propagate_const_binary` into `_commons`\n\nMove const-value propagation utilities to `_commons.py` alongside\n`atom_const_val`. Replace inline const propagation in reshape, slice,\ntranspose, and tile with `propagate_const_unary` calls using `partial`,\n`itemgetter`, or named closures. Remove unnecessary `np.asarray` and\n`.ravel()` round-trips \u2014 const values are stored in natural shape.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* docs: remove resolved items from TODO.md\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* refactor(interpret): make `propagate_const_binary` take a callable transform\n\nSymmetric with `propagate_const_unary`. The ufunc dict lookup\nmoves from `_commons.py` into `propagate_const_elementwise` in\n`_elementwise.py`, keeping the shared utility interface clean.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* refactor(interpret): extract `copy_index_sets` utility into `_commons`\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* docs(interpret): explain downstream impact in `propagate_const_*` docstrings\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* docs(interpret): document conservative fallback invariant for const_vals\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-20T15:52:27+01:00",
          "tree_id": "1fb12c5b3c013bef6a21f2d9113078ec71bdc9c8",
          "url": "https://github.com/adrhill/asdex/commit/18f12d46d7e0b4f42958d5d784c84b2f55d6557c"
        },
        "date": 1771599190062,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 792.37178871351,
            "unit": "iter/sec",
            "range": "stddev: 0.0033379364365099056",
            "extra": "mean: 1.2620338258427826 msec\nrounds: 178"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3306.331088276839,
            "unit": "iter/sec",
            "range": "stddev: 0.000006561970352595943",
            "extra": "mean: 302.45004910296814 usec\nrounds: 2118"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 39.199111967639055,
            "unit": "iter/sec",
            "range": "stddev: 0.019580880003549957",
            "extra": "mean: 25.51078199999921 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 83.15790203723284,
            "unit": "iter/sec",
            "range": "stddev: 0.00040215842762637903",
            "extra": "mean: 12.02531539999967 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 21.033269737508014,
            "unit": "iter/sec",
            "range": "stddev: 0.016096975600604108",
            "extra": "mean: 47.54372536842093 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 243.41569654513728,
            "unit": "iter/sec",
            "range": "stddev: 0.0003605956600528783",
            "extra": "mean: 4.108198502369658 msec\nrounds: 211"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 30.34087199022872,
            "unit": "iter/sec",
            "range": "stddev: 0.0008961091991275546",
            "extra": "mean: 32.958841800000016 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 10.417753650741876,
            "unit": "iter/sec",
            "range": "stddev: 0.022684410072598304",
            "extra": "mean: 95.9899833999998 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 90.22501554588302,
            "unit": "iter/sec",
            "range": "stddev: 0.012057938986435375",
            "extra": "mean: 11.08340069491548 msec\nrounds: 59"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3280.8559837031894,
            "unit": "iter/sec",
            "range": "stddev: 0.000032222706948445385",
            "extra": "mean: 304.79850531911296 usec\nrounds: 2256"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 24.80666146973708,
            "unit": "iter/sec",
            "range": "stddev: 0.001958743692824652",
            "extra": "mean: 40.31175260000026 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 19.326580007827673,
            "unit": "iter/sec",
            "range": "stddev: 0.013885459815837517",
            "extra": "mean: 51.742212000000976 msec\nrounds: 20"
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
          "id": "67fbae7d34492dfafdb27279f2c4e94ee3437dfb",
          "message": "refactor(interpret): introduce `IndexSet` abstraction (#60)\n\n* refactor(interpret): introduce `IndexSet` abstraction for future backend swap\n\nAdd `IndexSet` type alias and factory helpers (`empty_index_set`,\n`singleton_index_set`, `empty_index_sets`, `identity_index_sets`) to\ncentralize index set construction in `_commons.py`.\n\nUpdate all handlers, `detection.py`, and tests to use the new helpers\ninstead of bare `set()` / `{i}` / `[set() for ...]` construction.\nThis confines the `set[int]` dependency to the helpers, enabling a\nfuture swap to `pyroaring.BitMap` by changing only the factory functions.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* refactor(interpret): remove `IndexSets` type alias\n\nReplace `IndexSets` with `list[IndexSet]` everywhere to avoid\nthe visually similar `IndexSet` / `IndexSets` pair.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-20T17:59:00+01:00",
          "tree_id": "e8dee067a23c8ac10d12777e6248158111f15482",
          "url": "https://github.com/adrhill/asdex/commit/67fbae7d34492dfafdb27279f2c4e94ee3437dfb"
        },
        "date": 1771606782243,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 816.7046193432479,
            "unit": "iter/sec",
            "range": "stddev: 0.002809837545300645",
            "extra": "mean: 1.2244328932584572 msec\nrounds: 178"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3301.02072356661,
            "unit": "iter/sec",
            "range": "stddev: 0.00000907544324414739",
            "extra": "mean: 302.93660165803 usec\nrounds: 2051"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 59.38721045586209,
            "unit": "iter/sec",
            "range": "stddev: 0.0004822521313502439",
            "extra": "mean: 16.838642400003323 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 83.79039316998548,
            "unit": "iter/sec",
            "range": "stddev: 0.00027866948615172293",
            "extra": "mean: 11.934542400001646 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 21.229325359203813,
            "unit": "iter/sec",
            "range": "stddev: 0.015430402982792546",
            "extra": "mean: 47.104652789470656 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 251.42000727762192,
            "unit": "iter/sec",
            "range": "stddev: 0.00003213375204018758",
            "extra": "mean: 3.9774082056078544 msec\nrounds: 214"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 30.51636894367823,
            "unit": "iter/sec",
            "range": "stddev: 0.0014175528246474227",
            "extra": "mean: 32.76929840000378 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 11.626124367429021,
            "unit": "iter/sec",
            "range": "stddev: 0.0010791068421252767",
            "extra": "mean: 86.01318619999745 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 120.27343172451452,
            "unit": "iter/sec",
            "range": "stddev: 0.00012254737225083593",
            "extra": "mean: 8.314388187496746 msec\nrounds: 16"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3326.5003255073534,
            "unit": "iter/sec",
            "range": "stddev: 0.000006836910326655765",
            "extra": "mean: 300.6162339237052 usec\nrounds: 2146"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 19.923974972953214,
            "unit": "iter/sec",
            "range": "stddev: 0.021347409167883247",
            "extra": "mean: 50.190787799999725 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 20.232329027935187,
            "unit": "iter/sec",
            "range": "stddev: 0.0008047236874152402",
            "extra": "mean: 49.425847050000016 msec\nrounds: 20"
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
          "id": "2032c886c215412736549a4a88861f6400d8b2a9",
          "message": "perf(interpret): remove unnecessary index set copies (#61)\n\n* perf(interpret): remove unnecessary index set copies\n\nIndex sets stored in `deps` are never mutated by subsequent handlers,\nso defensive `.copy()` calls in `permute_indices`, `conservative_indices`,\nbroadcast, pad, sort, top_k, dynamic_update_slice, and scan tiling\nare unnecessary. Removing them avoids millions of small-set copies.\n\nThe only place that mutates stored sets is `fixed_point_loop` (`|=`\non carry), which now copies carry sets on entry to avoid aliasing bugs.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* perf(interpret): use pyroaring for index sets\n\nSwap the `IndexSet` backend from Python's built-in `set` to Roaring\nbitmaps (`pyroaring.BitMap`). Only `_commons.py` changes \u2014 all ~28\nhandler modules already use the factory helpers and compatible ops\n(`|=`, `.copy()`, `|`, `len()`, `bool()`, iteration).\n\nFix bare `set()` in `_top_k.py` to use `empty_index_sets()` factory.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Revert \"perf(interpret): use pyroaring for index sets\"\n\nThis reverts commit 9a3f497052b6f760f6dafaf448501c84dc89c8fd.\n\n* docs(interpret): update IndexSet docstring after pyroaring benchmark\n\npyroaring.BitMap and int bitmasks were benchmarked against set[int].\nset[int] wins for the typical workload (small sparse sets, large universe).\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* docs(interpret): document index set aliasing invariant\n\nHandlers must not mutate sets from `deps` since copies were removed\nfor performance. The only exception is `fixed_point_loop`, which\nexplicitly copies carry sets before in-place mutation.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-20T22:52:24+01:00",
          "tree_id": "553bbfd621d92cca5e919de63d918a5850616fa3",
          "url": "https://github.com/adrhill/asdex/commit/2032c886c215412736549a4a88861f6400d8b2a9"
        },
        "date": 1771624385035,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 850.8536821765543,
            "unit": "iter/sec",
            "range": "stddev: 0.0029358324666130566",
            "extra": "mean: 1.1752902067038329 msec\nrounds: 179"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3347.534544134219,
            "unit": "iter/sec",
            "range": "stddev: 0.000007742304312369538",
            "extra": "mean: 298.727313136251 usec\nrounds: 2162"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 60.7821239481678,
            "unit": "iter/sec",
            "range": "stddev: 0.00047425032540042834",
            "extra": "mean: 16.45220560000098 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 81.64201423446232,
            "unit": "iter/sec",
            "range": "stddev: 0.00021792935695950805",
            "extra": "mean: 12.248595400012618 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 20.56787633773195,
            "unit": "iter/sec",
            "range": "stddev: 0.01830257317128752",
            "extra": "mean: 48.619506631586034 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 248.69138563686113,
            "unit": "iter/sec",
            "range": "stddev: 0.000028911510398288377",
            "extra": "mean: 4.02104800469526 msec\nrounds: 213"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 29.244772410640632,
            "unit": "iter/sec",
            "range": "stddev: 0.0004495523208422888",
            "extra": "mean: 34.19414540002208 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 10.197554282213167,
            "unit": "iter/sec",
            "range": "stddev: 0.025385138541575928",
            "extra": "mean: 98.06272880000506 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 96.31658397899372,
            "unit": "iter/sec",
            "range": "stddev: 0.010445384294121231",
            "extra": "mean: 10.38242801694562 msec\nrounds: 59"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3332.4214939902536,
            "unit": "iter/sec",
            "range": "stddev: 0.000013071366817416024",
            "extra": "mean: 300.0820879961965 usec\nrounds: 2216"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 25.346947753501585,
            "unit": "iter/sec",
            "range": "stddev: 0.001183554105003659",
            "extra": "mean: 39.45248200000151 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 19.385802751913477,
            "unit": "iter/sec",
            "range": "stddev: 0.013582004569320702",
            "extra": "mean: 51.58414189999405 msec\nrounds: 20"
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
          "id": "c47a9456cac794f2e64c28c3dd8ddeba3edf0f26",
          "message": "docs: remove complexity claims\n\nThe O(mn) and O(n\u00b2) claims for verification assumed T(f) = O(max(m,n)),\nwhich isn't guaranteed. The O(|V| + |E|) claim for greedy coloring\ndidn't account for star coloring's more expensive neighbor-of-neighbor\nchecks.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-21T15:33:43+01:00",
          "tree_id": "8c4dffd2cfc0c221af2f8a3760f65183720ae595",
          "url": "https://github.com/adrhill/asdex/commit/c47a9456cac794f2e64c28c3dd8ddeba3edf0f26"
        },
        "date": 1771684487377,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 844.7940319736151,
            "unit": "iter/sec",
            "range": "stddev: 0.0029504359019137945",
            "extra": "mean: 1.1837204835169008 msec\nrounds: 182"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3110.998753660923,
            "unit": "iter/sec",
            "range": "stddev: 0.00006596238272598864",
            "extra": "mean: 321.4401802068331 usec\nrounds: 2031"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 61.338533401547195,
            "unit": "iter/sec",
            "range": "stddev: 0.000457191651394767",
            "extra": "mean: 16.30296560000204 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 84.62813247649675,
            "unit": "iter/sec",
            "range": "stddev: 0.0003415681688527455",
            "extra": "mean: 11.816401599996595 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 24.030393358213786,
            "unit": "iter/sec",
            "range": "stddev: 0.0005526983680647108",
            "extra": "mean: 41.61396715789473 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 250.54109458335913,
            "unit": "iter/sec",
            "range": "stddev: 0.00002066550859356318",
            "extra": "mean: 3.991361184331713 msec\nrounds: 217"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 30.955694346720144,
            "unit": "iter/sec",
            "range": "stddev: 0.000647209152151131",
            "extra": "mean: 32.30423419999795 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 11.253434016337211,
            "unit": "iter/sec",
            "range": "stddev: 0.0005734415960136834",
            "extra": "mean: 88.86176419999856 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 122.36042346496323,
            "unit": "iter/sec",
            "range": "stddev: 0.00016631014597554748",
            "extra": "mean: 8.172577142856495 msec\nrounds: 14"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3305.2895672414784,
            "unit": "iter/sec",
            "range": "stddev: 0.000007218132690236099",
            "extra": "mean: 302.5453533363426 usec\nrounds: 2233"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 24.547270998876815,
            "unit": "iter/sec",
            "range": "stddev: 0.0014519091322791314",
            "extra": "mean: 40.73772600000041 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 18.63393102124747,
            "unit": "iter/sec",
            "range": "stddev: 0.01535837268428393",
            "extra": "mean: 53.665541578947725 msec\nrounds: 19"
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
          "id": "047a2b185b2f53e55b204f7a1937b48a228371ec",
          "message": "feat(verify): add randomized matrix-vector product verification (#62)\n\n* feat(verify): add randomized matvec verification and AD mode selection\n\nReplace the O(n\u00b2) dense-only verification with a cheap O(k) randomized\nmatvec default path. Both `check_jacobian_correctness` and\n`check_hessian_correctness` gain `method`, `ad_mode`, `num_probes`, and\n`seed` parameters.\n\n- `method=\"matvec\"` (default) checks via randomized matrix-vector\n  products against `jax.jvp` / HVP references\n- `method=\"dense\"` preserves the original full-matrix comparison\n- `ad_mode` controls the reference AD mode for both paths:\n  Jacobian accepts \"forward\" / \"reverse\",\n  Hessian accepts \"fwd_over_rev\" / \"rev_over_fwd\" / \"rev_over_rev\"\n- Tolerances default to 1e-5 (matvec) or 1e-7 (dense)\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* feat(verify): auto-select forward/reverse based on Jacobian shape\n\nWhen `ad_mode` is not specified, pick the mode that maximizes\ndetection power per probe: forward (JVP) when m >= n,\nreverse (VJP) when m < n.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* docs: update verification how-to guides for new API\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-21T17:54:05+01:00",
          "tree_id": "644803c127208018560db850614293401d23b5dc",
          "url": "https://github.com/adrhill/asdex/commit/047a2b185b2f53e55b204f7a1937b48a228371ec"
        },
        "date": 1771692883596,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 858.2120155152836,
            "unit": "iter/sec",
            "range": "stddev: 0.00284591083532933",
            "extra": "mean: 1.1652132362649161 msec\nrounds: 182"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3342.9018981078243,
            "unit": "iter/sec",
            "range": "stddev: 0.000008635629963343361",
            "extra": "mean: 299.14129414507437 usec\nrounds: 2152"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 58.330897532256415,
            "unit": "iter/sec",
            "range": "stddev: 0.0004313517798775312",
            "extra": "mean: 17.143573000004153 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 80.46179405274155,
            "unit": "iter/sec",
            "range": "stddev: 0.00020676027131089758",
            "extra": "mean: 12.428258800002823 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 23.51532895109647,
            "unit": "iter/sec",
            "range": "stddev: 0.0005572506799835429",
            "extra": "mean: 42.52545231579132 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 249.30826799805723,
            "unit": "iter/sec",
            "range": "stddev: 0.00002790900360007633",
            "extra": "mean: 4.011098420561779 msec\nrounds: 214"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 30.098099457922675,
            "unit": "iter/sec",
            "range": "stddev: 0.0005043319718440923",
            "extra": "mean: 33.22468919999437 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 11.497921265517627,
            "unit": "iter/sec",
            "range": "stddev: 0.0010883339343599088",
            "extra": "mean: 86.97224280001024 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 123.21532828263939,
            "unit": "iter/sec",
            "range": "stddev: 0.0001670011528503709",
            "extra": "mean: 8.115873357137307 msec\nrounds: 14"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3361.375335158679,
            "unit": "iter/sec",
            "range": "stddev: 0.000012717556775682645",
            "extra": "mean: 297.4972742675859 usec\nrounds: 2184"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 25.35023350252607,
            "unit": "iter/sec",
            "range": "stddev: 0.0005257201125525064",
            "extra": "mean: 39.447368399993366 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 19.097500883274197,
            "unit": "iter/sec",
            "range": "stddev: 0.014398258752199822",
            "extra": "mean: 52.36287229999874 msec\nrounds: 20"
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
          "id": "d11498e51cc45cde5c5d4b8998485e824808f573",
          "message": "feat!: unify AD mode API (#63)\n\n* feat!: unify AD mode API\n\nConsistent `mode` parameter everywhere, replacing the mix of\n`partition`, `hvp_mode`, and `ad_mode`. Rename `HvpMode` to\n`HessianMode` and introduce `JacobianMode` with `\"fwd\"`/`\"rev\"`\nvalues (matching JAX's `jacfwd`/`jacrev` naming). Add validation\nfor unknown mode values in `color_jacobian_pattern`.\n\nBREAKING CHANGE: `partition=\"row\"/\"column\"` is now `mode=\"rev\"/\"fwd\"`,\n`hvp_mode` is now `mode`, `ad_mode` is now `mode`,\n`HvpMode` is renamed to `HessianMode`.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* feat!: separate coloring mode from AD mode\n\nRename `ColoringMode` to domain-appropriate values\n(`\"row\"`, `\"column\"`, `\"symmetric\"`) and add `\"auto\"` to all mode types.\n\nReplace the single `mode` kwarg on `jacobian()` and `hessian()` with\nseparate `coloring_mode` and `ad_mode` kwargs, decoupling the coloring\nstrategy from the AD primitive selection.\n\nKey changes:\n- `modes.py`: add `resolve_ad_mode()` and `resolve_hessian_mode()`\n- `coloring.py`: `color_jacobian_pattern` accepts `\"symmetric\"`\n- `decompression.py`: `hessian()` accepts `coloring_mode` for\n  row/column coloring with HVPs\n- `verify.py`: rename `mode` \u2192 `ad_mode`\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* refactor(coloring): centralize mode validation and resolution\n\nAdd `assert_coloring_mode`, `assert_jacobian_mode`, `assert_hessian_mode`\nvalidators and `resolve_coloring_mode` to `modes.py` as single source of truth.\n\n- Derive valid mode sets from `Literal` types via `get_args`\n- Push coloring-from-AD resolution into `resolve_coloring_mode`\n- Push coloring dispatch into `color_hessian_pattern` (accepts `coloring_mode`)\n- Extract `_empty_colored_pattern` helper for zero-nnz patterns\n- Rename bare `mode` parameters to `coloring_mode` everywhere\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* refactor: exhaustive match/case with `assert_never`\n\nReplace `if`/`elif` chains on mode arguments with exhaustive\n`match`/`case` using `assert_never` for unreachable branches.\nValidate `coloring_mode` and `ad_mode` eagerly in `jacobian()`\nand `hessian()` so typos raise at construction time.\n\nSuppresses `ty` false positives where match/case narrowing on\nattribute access is not yet supported (`@Todo`).\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* docs: update for new `coloring_mode`/`ad_mode` API\n\nFix parameter names in how-to guides (`mode` \u2192 `coloring_mode`/`ad_mode`)\nand update reference pages to reflect the new public API:\nremove `color_rows`, `color_cols`, `color_symmetric` and add\n`ColoringMode`, `JacobianMode`, `HessianMode`.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* refactor(modes): centralize input validation and mark helpers private\n\nAdd `_assert_jacobian_args` and `_assert_hessian_args` to centralize\nthe shared validation (mode assertions + coloring_mode-ignored warning)\nused by `jacobian`, `hessian`, `check_jacobian_correctness`, and\n`check_hessian_correctness`.\n\nPrefix all helper functions in `modes.py` with underscore since only\nthe type aliases are part of the public API.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* test: fix stale mode terminology and add missing mode tests\n\nRename test names and docstrings that still referenced `hvp_mode` or\nbare `mode` to use the new `ad_mode`/`coloring_mode` API.  Add tests\nfor column+rev incompatibility, invalid mode validation, and the\ncoloring_mode-ignored warning.  Also include the minor modes.py\ndocstring fix.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* test: cover remaining reachable mode validation paths\n\nAdd tests for:\n- Hessian coloring_mode-ignored warning (_assert_hessian_args)\n- _resolve_ad_mode raising on unresolved coloring_mode=\"auto\"\n- Empty non-square pattern with symmetric coloring (_empty_colored_pattern)\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* test: exclude unreachable branches from coverage\n\nExclude `assert_never()`, `case _ as unreachable:`, and\n`if TYPE_CHECKING:` lines from coverage reports via\n`[tool.coverage.report]` in pyproject.toml.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-21T21:48:17+01:00",
          "tree_id": "5e28607067564842f4e4033909344a1dbf9acb23",
          "url": "https://github.com/adrhill/asdex/commit/d11498e51cc45cde5c5d4b8998485e824808f573"
        },
        "date": 1771706937208,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 840.3929546409895,
            "unit": "iter/sec",
            "range": "stddev: 0.0030180598920961886",
            "extra": "mean: 1.1899195423731195 msec\nrounds: 177"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3253.506438144869,
            "unit": "iter/sec",
            "range": "stddev: 0.000039471492691002934",
            "extra": "mean: 307.3606949953338 usec\nrounds: 2118"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 59.734561590613104,
            "unit": "iter/sec",
            "range": "stddev: 0.0005078685265224629",
            "extra": "mean: 16.740727199999128 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 77.11610446455897,
            "unit": "iter/sec",
            "range": "stddev: 0.0014559462925232151",
            "extra": "mean: 12.96746000000013 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 20.51922044937148,
            "unit": "iter/sec",
            "range": "stddev: 0.015784789925303182",
            "extra": "mean: 48.7347948947364 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 250.74193693615496,
            "unit": "iter/sec",
            "range": "stddev: 0.00004433561243032252",
            "extra": "mean: 3.988164134883526 msec\nrounds: 215"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 30.520557344200146,
            "unit": "iter/sec",
            "range": "stddev: 0.00034150235812334856",
            "extra": "mean: 32.764801399999044 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 9.859623667195766,
            "unit": "iter/sec",
            "range": "stddev: 0.031092635241702502",
            "extra": "mean: 101.4237494000028 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 96.25429082543451,
            "unit": "iter/sec",
            "range": "stddev: 0.011225842022172068",
            "extra": "mean: 10.38914724137947 msec\nrounds: 58"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3280.0620544069498,
            "unit": "iter/sec",
            "range": "stddev: 0.000009027701532782302",
            "extra": "mean: 304.87228089372366 usec\nrounds: 2193"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 25.13158937896158,
            "unit": "iter/sec",
            "range": "stddev: 0.0007454069748235143",
            "extra": "mean: 39.7905594000008 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 18.641544443978404,
            "unit": "iter/sec",
            "range": "stddev: 0.01646206413038489",
            "extra": "mean: 53.643623950000574 msec\nrounds: 20"
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
          "id": "89a23cb80924fe0f850ffd27f93f7a6e49fc73e3",
          "message": "refactor!(API): require pre-computed coloring (#64)\n\n* feat!: require `colored_pattern` or `input_shape` in `jacobian()`/`hessian()`\n\n`jacobian(f)(x)` and `hessian(f)(x)` previously re-detected sparsity\nand re-colored on every call. Now either a pre-computed `colored_pattern`\nor an `input_shape` must be provided, and detection + coloring happen\nonce at definition time.\n\n- Add `input_shape` kwarg to `jacobian()` and `hessian()`\n- Add mutual exclusivity check in `_assert_jacobian_args`/`_assert_hessian_args`\n- Remove `colored_pattern is None` branch from `_eval_jacobian`/`_eval_hessian`\n- Update all tests and docs\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* feat!: freeze AD mode in `ColoredPattern`, remove `ColoringMode`\n\nReplace `mode: ColoringMode` on `ColoredPattern` with `symmetric: bool`\n+ `mode: str` (the resolved AD mode, never \"auto\").\nThis eliminates `ColoringMode` from the public API entirely.\n\n`jacobian()` and `hessian()` now require a pre-computed `ColoredPattern`;\nthe `input_shape` convenience path is removed.\n\nKey changes:\n- `ColoredPattern` stores `symmetric` + `mode` (\"fwd\"/\"rev\"/HVP modes)\n- `ColoringMode` type alias removed from `modes.py` and `__init__.py`\n- `jacobian(f, colored_pattern)` / `hessian(f, colored_pattern)` simplified\n- `from_coordinates` renamed to `from_coo` (scipy convention)\n- All coloring functions use `symmetric=` + `mode=` keyword args\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* refactor: rename `colored_pattern` to `coloring`, `sparsity_pattern` to `sparsity`\n\nShorter, more consistent variable names throughout\nthe codebase, tests, and documentation.\n\n* refactor(coloring): put `mode` before `symmetric` in keyword args\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Update docs\n\n* feat!: add one-call `jacobian`/`hessian` API, rename old to `_from_coloring`\n\n`jacobian(f, input_shape)` and `hessian(f, input_shape)` now handle\ndetection, coloring, and decompression in one call.\nThe old `jacobian(f, coloring)` and `hessian(f, coloring)` are renamed\nto `jacobian_from_coloring` and `hessian_from_coloring`.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* test: use one-call `jacobian`/`hessian` API in tests\n\nSwitch ~35 end-to-end tests from verbose\n`jacobian_from_coloring(f, jacobian_coloring(f, shape))` to the new\n`jacobian(f, input_shape=shape)` (and likewise for `hessian`).\nTests that exercise explicit coloring paths stay with `from_coloring`.\n\nAlso parametrize `test_hessian_non_symmetric_coloring` over all three\nHessian AD modes.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* docs: minor changes to \"Getting started\" guide\n\n* docs: improve how-to guides for jacobians and hessians\n\n- Rewrite Basic Usage sections with cross-references to API docs\n- Add executable code blocks to Precomputing sections showing coloring output\n- Add imports to Saving/Loading code blocks for self-contained examples\n- Inline multi-dimensional input notes, remove standalone sections\n- Replace \"colored pattern\" with \"coloring\" throughout\n- Add one-call API `mode` parameter mention to Choosing Row vs Column Coloring\n- Fix `jnp` import order in Getting Started tutorial\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* refactor!: rename `color_*_pattern` to `*_coloring_from_sparsity`\n\nRename `color_jacobian_pattern` to `jacobian_coloring_from_sparsity`\nand `color_hessian_pattern` to `hessian_coloring_from_sparsity`\nfor consistency with the `*_coloring` naming convention.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* refactor!: make `coloring` mandatory in `check_*_correctness`\n\n`check_jacobian_correctness` and `check_hessian_correctness` now require\na pre-computed `coloring` as a positional argument instead of accepting\nan optional keyword. Remove auto-detection branches, unused imports, and\nthe non-existent `ad_mode` parameter from docs.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Minor tweaks to docstrings\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-02-22T01:21:19+01:00",
          "tree_id": "04c653e55f9fa29ab7878972af995167e180f88e",
          "url": "https://github.com/adrhill/asdex/commit/89a23cb80924fe0f850ffd27f93f7a6e49fc73e3"
        },
        "date": 1771719721006,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 794.767798100254,
            "unit": "iter/sec",
            "range": "stddev: 0.0037189376688620246",
            "extra": "mean: 1.2582291360952413 msec\nrounds: 169"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3337.9463102185696,
            "unit": "iter/sec",
            "range": "stddev: 0.000010757101011014845",
            "extra": "mean: 299.58540583431966 usec\nrounds: 1954"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 59.713199889458544,
            "unit": "iter/sec",
            "range": "stddev: 0.0005753496294292766",
            "extra": "mean: 16.746716000000106 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 103.69472389428726,
            "unit": "iter/sec",
            "range": "stddev: 0.0002659972911703389",
            "extra": "mean: 9.643692199995257 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 22.46058089440822,
            "unit": "iter/sec",
            "range": "stddev: 0.012884775017442611",
            "extra": "mean: 44.52244600000349 msec\nrounds: 19"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 250.1920289813675,
            "unit": "iter/sec",
            "range": "stddev: 0.000044750004852854335",
            "extra": "mean: 3.9969298944950515 msec\nrounds: 218"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 30.359460628471936,
            "unit": "iter/sec",
            "range": "stddev: 0.0005372516271521271",
            "extra": "mean: 32.93866160000789 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 27.22939960453755,
            "unit": "iter/sec",
            "range": "stddev: 0.0007653826011410219",
            "extra": "mean: 36.72501099999863 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 108.72873537756934,
            "unit": "iter/sec",
            "range": "stddev: 0.007540434363582972",
            "extra": "mean: 9.197200689655949 msec\nrounds: 58"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3363.0657876179353,
            "unit": "iter/sec",
            "range": "stddev: 0.000008840694381832143",
            "extra": "mean: 297.34773660443363 usec\nrounds: 2221"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 25.377193277854804,
            "unit": "iter/sec",
            "range": "stddev: 0.0005925186228893789",
            "extra": "mean: 39.405461000001196 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 25.5704144985585,
            "unit": "iter/sec",
            "range": "stddev: 0.0010615404415476647",
            "extra": "mean: 39.10769612500312 msec\nrounds: 24"
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
          "id": "87022d28acde3b083ed3501358df2dad61e24681",
          "message": "docs: update README",
          "timestamp": "2026-02-22T01:28:05+01:00",
          "tree_id": "1881e0201626189bc7c5d2a3025994364e607048",
          "url": "https://github.com/adrhill/asdex/commit/87022d28acde3b083ed3501358df2dad61e24681"
        },
        "date": 1771720130303,
        "tool": "pytest",
        "benches": [
          {
            "name": "tests/test_benchmarks.py::test_heat_detection",
            "value": 662.4741502201823,
            "unit": "iter/sec",
            "range": "stddev: 0.005329607505662316",
            "extra": "mean: 1.5094928604650255 msec\nrounds: 129"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_coloring",
            "value": 3283.5649657028052,
            "unit": "iter/sec",
            "range": "stddev: 0.000007633010419312407",
            "extra": "mean: 304.5470427553921 usec\nrounds: 2105"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_materialization",
            "value": 58.03020664917334,
            "unit": "iter/sec",
            "range": "stddev: 0.0004595149700283407",
            "extra": "mean: 17.232404599997153 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_heat_end_to_end",
            "value": 100.27799667907357,
            "unit": "iter/sec",
            "range": "stddev: 0.00046881559437838926",
            "extra": "mean: 9.972277399999996 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_detection",
            "value": 19.35474301816279,
            "unit": "iter/sec",
            "range": "stddev: 0.023258211307440387",
            "extra": "mean: 51.66692211111171 msec\nrounds: 18"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_coloring",
            "value": 246.6799297119202,
            "unit": "iter/sec",
            "range": "stddev: 0.000060774692158124454",
            "extra": "mean: 4.053836082926683 msec\nrounds: 205"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_materialization",
            "value": 30.02217485870521,
            "unit": "iter/sec",
            "range": "stddev: 0.0008709241216604976",
            "extra": "mean: 33.30871280000025 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_convnet_end_to_end",
            "value": 27.316013767774592,
            "unit": "iter/sec",
            "range": "stddev: 0.0005344329204518354",
            "extra": "mean: 36.608562599998606 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_detection",
            "value": 93.63387585238205,
            "unit": "iter/sec",
            "range": "stddev: 0.012930935828396764",
            "extra": "mean: 10.67989539999972 msec\nrounds: 55"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_coloring",
            "value": 3274.8477902757645,
            "unit": "iter/sec",
            "range": "stddev: 0.000009415037798966306",
            "extra": "mean: 305.35770333185263 usec\nrounds: 2191"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_materialization",
            "value": 24.87003664948106,
            "unit": "iter/sec",
            "range": "stddev: 0.0005747262250399623",
            "extra": "mean: 40.20902799999959 msec\nrounds: 5"
          },
          {
            "name": "tests/test_benchmarks.py::test_rosenbrock_end_to_end",
            "value": 25.62462762193664,
            "unit": "iter/sec",
            "range": "stddev: 0.0007321251787339029",
            "extra": "mean: 39.024957347825946 msec\nrounds: 23"
          }
        ]
      }
    ]
  }
}