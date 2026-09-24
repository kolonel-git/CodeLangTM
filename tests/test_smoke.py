import numpy as np
import pytest

import codelangtm
from codelangtm.cli import main
from codelangtm.features import Binarizer, expand_literals


def test_version():
    assert codelangtm.__version__


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0
    assert codelangtm.__version__ in capsys.readouterr().out


def test_binarizer_and_literals():
    snippets = ["def f():\n    return 1", "int main() { return 0; }"]
    b = Binarizer(n_features=50).fit(snippets)
    x = b.transform(snippets)
    assert x.shape == (2, len(b.vocabulary_))
    lit = expand_literals(x)
    assert lit.shape == (2, 2 * x.shape[1])
    assert np.all(lit[:, : x.shape[1]] + lit[:, x.shape[1] :] == 1)
