import pytest                                                  # test framework

from corpora import generate_fuzz                              # deterministic noisy-input generator
from shared.constants import EMOTION_LABELS                    # the 28 valid labels
import thresholds as T                                         # FUZZ_N (battery size)

pytestmark = pytest.mark.slow                                  # heavy (thousands of inputs) -> opt-in

_FUZZ = generate_fuzz(T.FUZZ_N)                                # generate the fuzz inputs once


@pytest.mark.parametrize("text", _FUZZ, ids=[f"f{i:04d}" for i in range(len(_FUZZ))])  # one test per input
def test_fuzz_no_crash(io, text):                             # noisy input -> no crash, valid label
    r, ok = io(text, valid=True)                              # must not raise; record validity verdict
    assert ok, (                                              # only validity is asserted
        f"non-canonical label {r['final_label']!r} for input {text[:40]!r}")
