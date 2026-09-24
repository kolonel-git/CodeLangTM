import pytest

from codelangtm.data import Snippet
from codelangtm.labels import (
    HTML_TAG,
    check_label,
    filter_labels,
    language_from_path,
    markup_share,
    template_fraction,
)

PY = "import os\n\ndef f(x):\n    return x + 1\n"
CPP = "#include <iostream>\nclass A {};\nint main() { std::cout << 1; }\n"
C = "#include <stdio.h>\nint main(void) { printf(\"x\"); return 0; }\n"


def snip(text, language, path, source="github"):
    return Snippet(text, language, source, "o/r", "c", path, "MIT", 1, text.count("\n") + 1)


def test_language_from_path():
    assert language_from_path("a/b/main.PY") == "python"
    assert language_from_path("x.tsx") == "typescript"
    assert language_from_path("README") is None
    assert language_from_path("x.h", CPP) == "cpp"
    assert language_from_path("x.h", C) == "c"


def test_good_labels_pass():
    assert check_label(snip(PY, "python", "a.py")).ok
    assert check_label(snip(CPP, "cpp", "a.hpp")).ok
    assert check_label(snip(CPP, "cpp", "a.h")).ok


def test_extension_mismatch():
    r = check_label(snip(PY, "javascript", "a.py"))
    assert not r.ok
    assert any("extension implies" in x for x in r.reasons)


def test_header_c_vs_cpp():
    assert not check_label(snip(C, "cpp", "a.h")).ok  # plain C header labelled C++


def test_red_flag_other_language():
    r = check_label(snip("#include <stdio.h>\nx = 1\n", "python", "a.py"))
    assert not r.ok
    assert "content looks like another language" in r.reasons


def test_typescript_syntax_in_js():
    assert not check_label(snip("function f(a: string, b) {}\n", "javascript", "a.js")).ok


def test_minified_and_binary():
    assert not check_label(snip("var a=" + "1," * 400 + "\n", "javascript", "a.js")).ok
    assert not check_label(snip("x\x00y\n", "python", "a.py")).ok


def test_required_markers():
    assert not check_label(snip("just some words\nhere\n", "html", "a.html")).ok
    assert check_label(snip("<div>hi</div>\n", "html", "a.html")).ok
    assert not check_label(snip("hello world\n", "sql", "a.sql")).ok
    assert check_label(snip("SELECT 1 FROM t;\n", "sql", "a.sql")).ok


DBT = "{% macro m() %}\n{{ config(x=1) }}\nselect *\nfrom {{ ref('a') }}\n{% endmacro %}\n"
DBT_LIGHT = "select id,\n  name,\n  email\nfrom {{ ref('users') }}\nwhere active\n" * 2
JEKYLL = "---\nlayout: page\n---\n<div>{{ page.title }}</div>\n<p>{% include x.html %}</p>\n"


def test_template_fraction():
    assert template_fraction(DBT) == pytest.approx(4 / 5)
    assert template_fraction("---\nselect 1;\n") == 0.0  # `---` is a SQL comment
    assert template_fraction("---\n<p>x</p>\n", front_matter=True) == 0.5


def test_template_heavy_dropped():
    assert "template-heavy" in check_label(snip(DBT, "sql", "m.sql")).reasons
    assert "template-heavy" in check_label(snip(JEKYLL, "html", "i.html")).reasons


def test_lightly_templated_kept():
    assert check_label(snip(DBT_LIGHT, "sql", "m.sql")).ok  # 2 of 10 lines


def test_template_check_only_for_sql_and_html():
    rust = "fn main() {\n    println!(\"{{}}\", 1);\n    let v = vec![{{ 1 }}];\n}\n"
    assert check_label(snip(rust, "rust", "a.rs")).ok


SCRIPT_ONLY = "    var n = 0;\n    for (var i = 0; i < elements.length; i++) {\n" \
    "        n += elements[i].value;\n    }\n    return n;\n"


def test_html_tag_pattern():
    for tag in ("<div>", "</p>", "<my-el attr>", "<br/>", "<!-- c -->", "<!DOCTYPE html>", "<a"):
        assert HTML_TAG.search(tag), tag
    for not_tag in ("i < elements.length", "a<b", "x <= 3", "if (a<b.c)"):
        assert not HTML_TAG.search(not_tag), not_tag


def test_markup_share():
    assert markup_share("<div>\n  text\n</div>\n\n") == pytest.approx(2 / 3)
    assert markup_share(SCRIPT_ONLY) == 0.0


def test_script_only_html_dropped():
    reasons = check_label(snip(SCRIPT_ONLY, "html", "a.html")).reasons
    assert "expected markers missing" in reasons  # comparisons no longer count as tags


def test_mostly_script_html_dropped():
    text = "<script>\n" + SCRIPT_ONLY * 3 + "</script>\n"  # 2 tag lines of 17
    assert "mostly embedded script/style" in check_label(snip(text, "html", "a.html")).reasons


def test_html_with_some_markup_kept():
    text = "<div id=\"x\">\n<script>\n" + SCRIPT_ONLY + "</script>\n</div>\n"  # 4 of 9 lines
    assert check_label(snip(text, "html", "a.html")).ok


def test_wild_skips_extension_check():
    assert check_label(snip(PY, "python", "so-answer-123.txt", source="wild")).ok


def test_filter_labels_counts():
    items = [
        snip(PY, "python", "a.py"),
        snip(PY, "javascript", "a.py"),
        snip("x\x00", "go", "a.go"),
    ]
    kept, dropped = filter_labels(items)
    assert kept == [items[0]]
    assert sum(dropped.values()) >= 2


@pytest.mark.parametrize(
    ("language", "path", "text"),
    [
        ("go", "a.go", "package main\n\nfunc main() {\n\tprintln(1)\n}\n"),
        ("rust", "a.rs", "fn main() {\n    println!(\"x\");\n}\n"),
        ("java", "A.java", "public class A {\n    void f() {}\n}\n"),
    ],
)
def test_clean_code_passes(language, path, text):
    assert check_label(snip(text, language, path)).ok
