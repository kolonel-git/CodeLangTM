import pytest

from codelangtm.data import Snippet
from codelangtm.labels import check_label
from codelangtm.syntax import SYNTAX, is_cut, safe_line_starts, scan
from codelangtm.windows import extract_windows


def outside(text, language):
    return scan(text.splitlines(), SYNTAX[language]).outside


def test_c_block_comment_spans_lines():
    text = "int a;\n/* start\n * middle\n */\nint b;\n"
    assert outside(text, "cpp") == [True, True, False, False, True, True]


def test_comment_markers_inside_strings_and_line_comments_ignored():
    text = 'char* s = "/* not a comment";\n// also not /* one\nint x;\n'
    assert all(outside(text, "cpp"))


def test_python_docstring():
    text = 'def f():\n    """Doc\n    more\n    """\n    return 1\n'
    assert outside(text, "python") == [True, True, False, False, True, True]


def test_python_hash_inside_string_and_escaped_quote():
    assert all(outside("x = '# no' + \"a \\\" b\"\ny = 1\n", "python"))


def test_html_comment():
    assert outside("<p>a</p>\n<!-- a\nb -->\n<p>c</p>\n", "html") == [True, True, False, True, True]


def test_sql_line_comment_with_apostrophe():
    assert all(outside("-- don't break\nselect 'x' from t;\n", "sql"))


@pytest.mark.parametrize("language", ["go", "javascript"])
def test_backtick_strings_span_lines(language):
    text = "x := `line1\n/* not a comment */\nline3`\ny := 1\n"
    assert outside(text, language) == [True, False, False, True, True]


def test_unknown_language_unconstrained():
    assert safe_line_starts(["a", "b"], "cobol") is None
    assert safe_line_starts(["a", "b"], None) is None
    assert not is_cut(" * dangling\n */\n", "cobol")


@pytest.mark.parametrize(
    ("text", "language"),
    [
        (" * describes the function\n * more prose\n */\nint f() { return 1; }\n", "cpp"),
        ("int f() { return 1; }\n/**\n * Docs for g that get cut\n", "java"),
        ("x = 1\ndef g():\n    \"\"\"Starts a docstring\n    that never ends\n", "python"),
        ("<p>a</p>\n<!-- open comment\n<p>b</p>\n", "html"),
        ("more words of an old comment\n-->\n<p>b</p>\n", "html"),
    ],
)
def test_is_cut_detects_split_comments(text, language):
    assert is_cut(text, language)


@pytest.mark.parametrize(
    ("text", "language"),
    [
        ("/* full */\nint f() { return 1; }\n", "cpp"),
        ("/**\n * Docs\n */\npublic void g() {}\n", "java"),
        ('def f():\n    """Doc."""\n    return 1\n', "python"),
        ("const re = /.*/;\nconst s = x.replace(/a*/, '');\n", "javascript"),
        ("select a -- trailing */ remark\nfrom t;\n", "sql"),
    ],
)
def test_is_cut_accepts_whole_comments(text, language):
    assert not is_cut(text, language)


def commented_file():
    parts = []
    for k in range(20):
        parts.append("/**\n" + "".join(f" * doc line {j} of block {k}\n" for j in range(6)) + " */")
        parts.append("".join(f"int f{k}_{j}(int x) {{ return x + {j}; }}\n" for j in range(5)))
    return "\n".join(parts)


def test_windows_never_cut_comments():
    text = commented_file()
    naive = extract_windows(text, seed=7)
    assert any(is_cut(w.text, "cpp") for w in naive)  # the bug this fixes
    safe = extract_windows(text, seed=7, language="cpp")
    assert safe
    for w in safe:
        assert not is_cut(w.text, "cpp")
        assert 20 <= w.end_line - w.start_line + 1 <= 50


def test_windows_skip_giant_comment():
    text = "/*\n" + " * x\n" * 100 + " */\n" + "int a = 1;\n" * 30
    for w in extract_windows(text, language="cpp", min_nonblank=5):
        assert not is_cut(w.text, "cpp")


def test_label_check_flags_cut_snippet():
    text = " * leftover comment\n */\nint main() { return 0; }\n"
    s = Snippet(text, "cpp", "github", "o/r", "c", "a.cpp", "MIT", 1, 3)
    assert "cut comment or string" in check_label(s).reasons
