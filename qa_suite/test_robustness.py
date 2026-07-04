import pytest

from corpora import generate_fuzz
import thresholds as T

pytestmark = pytest.mark.slow

_FUZZ = generate_fuzz(T.FUZZ_N)


@pytest.mark.parametrize("text", _FUZZ, ids=[f"f{i:04d}" for i in range(len(_FUZZ))])
def test_fuzz_no_crash(io, text):
    r, ok = io(text, valid=True)
    assert ok, (
        f"non-canonical label {r['final_label']!r} for input {text[:40]!r}")
