import pytest

from codelangtm.windows import extract_windows, is_low_signal


def code(n):
    return "\n".join(f"x{i} = {i}" for i in range(n)) + "\n"


def test_windows_sizes_and_contiguity():
    ws = extract_windows(code(200), seed=1)
    assert ws
    for w in ws:
        n = w.end_line - w.start_line + 1
        assert 20 <= n <= 50
        assert w.text.count("\n") == n
    for a, b in zip(ws, ws[1:], strict=False):
        assert b.start_line == a.end_line + 1  # non-overlapping, contiguous


def test_deterministic():
    assert extract_windows(code(200), seed=3) == extract_windows(code(200), seed=3)


def test_short_file_yields_nothing():
    assert extract_windows(code(19)) == []


def test_max_windows():
    assert len(extract_windows(code(500), max_windows=2)) == 2


def test_crlf_normalized():
    ws = extract_windows(code(30).replace("\n", "\r\n"), min_lines=20, max_lines=20)
    assert "\r" not in ws[0].text


def test_license_header_skipped():
    header = "\n".join(["// Copyright 2020 Someone"] + ["// text"] * 24)
    body = code(30)
    ws = extract_windows(header + "\n" + body, min_lines=20, max_lines=20)
    assert all("Copyright" not in w.text for w in ws)
    assert ws  # code after header still extracted


def test_low_signal_cases():
    assert is_low_signal(["", "", "a"] * 3)  # mostly blank
    assert is_low_signal([f"# c{i}" for i in range(20)])  # pure comments
    assert not is_low_signal([f"x = {i}" for i in range(20)])


def test_bad_args():
    with pytest.raises(ValueError):
        extract_windows("x", min_lines=10, max_lines=5)
