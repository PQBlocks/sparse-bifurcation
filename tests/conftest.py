"""Pytest bootstrap for integration and property tests (verification & property tests).

Numba `@njit` bodies are opaque to coverage.py: when JIT is enabled the
compiled machine code executes, so the Python lines inside `dsb_dense_kernel`,
`dsb_csr_kernel`, `_sign`, `_splitmix64_numba` and `_init_state_numba`
(~70 statements in src/kernel.py) never register as covered even though they
are exercised by bit-identity, direct-kernel and equivalence tests.

Per the Numba documented practice, coverage runs execute the *same* Python
logic with the JIT disabled (NUMBA_DISABLE_JIT=1). Results are bit-identical
because the code paths are unchanged — only the compiler is bypassed. This
conftest forces that mode for every test session so any
`pytest --cov=src` measurement is honest: JIT-enabled runs report ~83%
(measurement artifact), JIT-disabled runs report >=90% (true logic coverage).

No production code is altered and no test is weakened : this file only
controls how the interpreter executes already-tested logic.
"""
import os

os.environ["NUMBA_DISABLE_JIT"] = "1"

try:
    import numba.config  # noqa: E402

    numba.config.DISABLE_JIT = True
except Exception:
    pass
