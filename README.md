# demoniC-testsuite

The conformance corpus and verification harnesses for
[demoniC](https://github.com/GusFromSpace/demoniC), a tensor-first systems
language. The compiler repository holds the language; this repository holds
the machinery that proves it behaves: an executable example corpus with
in-language tests, backend-parity diffing, a differential fuzzer, an external
NumPy oracle, and a per-example conformance baseline.

## Layout

- `examples/` — the corpus. Every file is a runnable demoniC program; most
  carry `fn test_*() -> bool` assertions that `dmc test` executes on the
  interpreter and, where JIT-eligible, on the JIT with results compared.
- `tools/` — the harnesses:

| Tool | What it proves |
| --- | --- |
| `example_runner.py` | each example, gated individually against `tools/example_baseline.json` by its executable test results on both backends |
| `diff_backends.py` | interpreter and JIT produce identical output for every JIT-runnable example |
| `jit_probes.py` | curated edge cases the corpus does not reach, diffed across backends |
| `diff_fuzz.py` | generated well-typed programs, both backends run and diffed |
| `numpy_oracle.py` | tensor ops checked against an independent NumPy reimplementation (catches bugs shared by both backends) |
| `lint_dmc.py` | style/consistency lints over the corpus |

## Running the suite

Build `dmc` from the compiler repository, then point the harnesses at the
binary:

```bash
git clone https://github.com/GusFromSpace/demoniC
cargo build --release --manifest-path demoniC/compiler/Cargo.toml
export DMC="$PWD/demoniC/compiler/target/release/dmc"
```

```bash
"$DMC" test examples          # in-language ground truth (interpreter)
"$DMC" test --jit examples    # same tests on the JIT, results compared
python3 tools/example_runner.py --dmc "$DMC"
python3 tools/diff_backends.py --dmc "$DMC"
python3 tools/jit_probes.py --dmc "$DMC"
python3 tools/diff_fuzz.py --dmc "$DMC" --iters 2000
python3 tools/numpy_oracle.py --dmc "$DMC"   # needs numpy
python3 tools/lint_dmc.py
```

The interpreter is the reference semantics: any unexplained divergence
between the backends is a compiler bug, and the suite treats it as one.

## CI

CI rebuilds `dmc` from the compiler repository's current `main` and runs the
battery above — on every push and pull request here, and once a day on a
schedule. A scheduled failure with no change in this repository means the
compiler moved in a way the suite catches; that is the point of the
schedule.

## Adding tests

A corpus file is an ordinary demoniC program with `fn test_*() -> bool`
assertions. A new file must pass under the interpreter, agree with the JIT
wherever it is JIT-eligible, and be recorded in the baseline:

```bash
python3 tools/example_runner.py --dmc "$DMC" --accept-new
```

Commit the baseline change together with the new file.

## License

Apache-2.0, the same terms as the compiler repository.
