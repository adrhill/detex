window.BENCHMARK_DATA = {
  "lastUpdate": 1770589474014,
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
          "message": "Support `while_loop`, `cond`, and `dynamic_slice` primitives (#24)\n\n* Add brusselator sparsity demo with diffrax\n\nDemonstrates asdex on a realistic reaction-diffusion ODE: detects the\nJacobian sparsity of the brusselator RHS (768 nnz, 11 colors) and\nshows the expected failure on diffrax's `while` primitive.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Support while_loop, cond, dynamic_slice, and diffrax primitives\n\nAdd handlers for 10 new JAX primitives needed by diffrax's diffeqsolve:\n- while: fixed-point iteration over body jaxpr\n- cond: union output deps across all branch jaxprs\n- dynamic_slice / dynamic_update_slice: precise when starts are static\n- not: zero derivative\n- select_if_vmap, nonbatchable, unvmap_any, unvmap_max, pure_callback:\n  conservative fallback\n\nAlso fix select_n to be element-wise instead of globally conservative,\nwhich is necessary for correct sparsity through diffrax control flow.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Simplify dynamic_slice handlers, move select_if_vmap to conservative\n\nReplace manual stride loops with np.indices + np.ravel_multi_index,\nextract _resolve_starts helper. Move select_if_vmap back to the\nconservative fallback since it's a different primitive from select_n.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Fix const_vals propagation into nested jaxprs, clean up handlers\n\nForward const_vals from outer-scope atoms to inner jaxpr variables so\nthat gather/scatter inside cond, while_loop, jit, and custom_jvp can\nresolve static indices precisely instead of falling back to conservative.\n\n- Add seed_const_vals and forward_const_vals helpers to _commons.py\n- Apply both helpers in prop_cond, prop_while, prop_nested_jaxpr,\n  and prop_custom_call\n- Extract prop_broadcast_in_dim into _broadcast.py\n- Clean up select_n: remove dead scalar broadcast, rename branches→cases\n- Add JAX doc URLs to handler docstrings\n- Add scan/associative_scan as explicit TODO errors\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Remove examples/ directory\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
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
          "message": "Clean up interpreter module and tests (#25)\n\n* Add JAX doc URLs and improve handler docstrings\n\n- Add JAX doc URLs to all handler docstrings missing them\n- Add JAX doc URLs to test module docstrings\n- Fix parameter names to match JAX API (start_indices, scatter_indices)\n- Document precise-path conditions in gather and scatter\n- Document conv assumption (feature_group_count=1, batch_group_count=1)\n- Clarify reduce_sum vs jax.lax.reduce naming difference\n- Document reshape bug with dimensions parameter\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Fix reshape handler ignoring `dimensions` parameter\n\nWhen `dimensions` is not None, `jax.lax.reshape` transposes the input\naxes before reshaping (e.g. `ravel(order='F')` emits `dimensions=(1,0)`).\nThe handler previously passed deps through in the original flat order,\nproducing incorrect (not merely conservative) sparsity patterns.\n\nFix by building a flat index mapping via `np.arange().reshape().transpose()`,\nmirroring the actual element reordering JAX performs.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Add 3D test for reshape dimensions fix\n\nTest ravel(order='F') on a (2, 3, 4) tensor,\nwhich emits dimensions=(2, 1, 0) — a higher-rank permutation\nthan the 2D case already tested.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Extract shared utilities from `_interpret` handlers\n\nAdd `atom_shape`, `flat_to_coords`, and `conservative_deps` helpers to\n`_commons.py` to reduce duplication across handler modules.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Split `_indexing.py` into `_slice.py`, `_squeeze.py`, and `_reshape.py`\n\nEach handler now lives in its own module, consistent with the rest of\nthe `_interpret` package.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Split test files to match `_interpret` handler modules\n\nOne test file per handler: `_foo.py` → `test_foo.py`.\nMove `test_control_flow.py` → `_interpret/test_select.py`,\nrename `test_dynamic_indexing.py` → `test_dynamic_slice.py`,\nand move fallback/custom_call tests into `test_internals.py`.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
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
          "message": "Add precise handlers for all reduction primitives (#33)\n\n* Add precise `reduce_max` primitive handler\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Add precise `reduce_prod` primitive handler\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Add precise `reduce_min` primitive handler\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Add zero-derivative handlers for `reduce_and`, `reduce_or`, `reduce_xor`\n\nBitwise reductions have zero Jacobian, so they use the existing\n`prop_zero_derivative` — no separate handler files needed.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Consolidate reduction handlers into single `_reduce.py` module\n\nThe four reduction primitives (reduce_sum, reduce_max, reduce_min,\nreduce_prod) share identical sparsity structure, so they now share\na single `prop_reduce` handler. Tests are parametrized over the\nreduce function and shape/axes combinations.\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Parametrize bitwise reduction tests over reduce_and/or/xor\n\nCo-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>\n\n* Update SKILL and TODOs\n\n---------\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
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
      }
    ]
  }
}