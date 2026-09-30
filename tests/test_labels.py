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
    check = check_label(snip(text, "html", "a.html"))  # 7 of 9 lines are inside <script>: hard
    assert not check.ok and check.hard


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


# --- stretch set (Stage B, B-S1) ---------------------------------------------------------------

CPP_IN_C = "#include <stdio.h>\nnamespace app {\nint f(int x) { return x; }\n}\n"


@pytest.mark.parametrize(
    ("language", "path", "text"),
    [
        ("c", "a.c", C),
        ("c", "a.h", C),
        ("csharp", "A.cs", "using System;\n\nnamespace App\n{\n    public class A { }\n}\n"),
        ("typescript", "a.ts", "export function f(a: string): boolean {\n  return !!a;\n}\n"),
        ("kotlin", "A.kt", "package app\n\nfun main() {\n    println(\"x\")\n}\n"),
        ("php", "a.php", "<?php\nnamespace App;\n\nclass A {\n    public function f() {}\n}\n"),
        ("ruby", "a.rb", "class A\n  def f(x)\n    x + 1\n  end\nend\n"),
    ],
)
def test_stretch_clean_code_passes(language, path, text):
    assert check_label(snip(text, language, path)).ok


def test_extensions_map_to_every_stretch_language():
    from codelangtm import ALL_LANGUAGES
    from codelangtm.labels import EXTENSION_LANGUAGE

    assert set(EXTENSION_LANGUAGE.values()) == set(ALL_LANGUAGES)
    assert language_from_path("a.cs") == "csharp"
    assert language_from_path("a.kt") == "kotlin"
    assert language_from_path("a.rb") == "ruby"


def test_c_file_with_cpp_constructs_dropped():
    r = check_label(snip(CPP_IN_C, "c", "a.c"))
    assert not r.ok
    assert "content looks like another language" in r.reasons
    assert not check_label(snip("int x = std::max(1, 2);\n", "c", "a.c")).ok


def test_c_comment_mentioning_class_is_fine():
    text = "/*\n * class of problems\n */\nint f(void) { return 1; }\n"
    assert check_label(snip(text, "c", "a.c")).ok


def test_qt_translation_file_is_not_typescript():
    xml = '<?xml version="1.0" encoding="utf-8"?>\n<TS version="2.1">\n</TS>\n'
    assert not check_label(snip(xml, "typescript", "a.ts")).ok


def test_csharp_and_java_are_told_apart():
    assert not check_label(snip("using System;\nclass A {}\n", "java", "A.java")).ok
    assert not check_label(snip("import java.util.List;\nclass A {}\n", "csharp", "A.cs")).ok


def test_stretch_wrong_extension_dropped():
    assert not check_label(snip("class A\nend\n", "ruby", "a.py")).ok


# --- embedded-language policy (Stage B, B-S2) --------------------------------------------------

MARKUP = "".join(f'<p class="c{i}">line {i}</p>\n' for i in range(10))
JS = "".join(f"  var v{i} = compute({i}) + {i};\n" for i in range(10))


def test_embedded_share_lives_in_labels():
    from codelangtm.labels import embedded_share

    assert embedded_share("<div>\n<script>\nvar a;\n</script>\n<p>") == 3 / 5
    assert embedded_share("var a;\nb();\n</script>\n<div>") == 3 / 4  # window starts inside


def test_html_over_half_embedded_is_hard():
    text = MARKUP + "<script>\n" + JS * 2 + "</script>\n"  # 22 of 32 lines inside
    check = check_label(snip(text, "html", "a.html"))
    assert not check.ok and check.hard
    assert check.reasons == ("mostly embedded script/style",)


def test_html_at_most_half_embedded_is_kept():
    text = MARKUP * 2 + "<script>\n" + JS + "</script>\n"  # 12 of 32 lines inside
    check = check_label(snip(text, "html", "a.html"))
    assert check.ok and not check.hard


def test_exactly_half_is_kept_over_half_is_not():
    just_over = "<p>a</p>\n<script>\nx();\n</script>\n"  # 3 of 4 lines inside
    exactly = "<p>a</p>\n<p>b</p>\n<script>x();\n</script>\n"  # 2 of 4 inside
    assert check_label(snip(exactly, "html", "a.html")).ok
    assert check_label(snip(just_over, "html", "a.html")).hard


def test_sparse_markup_without_embedded_code_is_dropped_not_hard():
    text = "<p>x</p>\n" + "plain prose line\n" * 30
    check = check_label(snip(text, "html", "a.html"))
    assert not check.ok and not check.hard
    assert check.reasons == ("too little markup",)


def test_hard_only_when_embedded_is_the_only_reason():
    text = "{% block a %}\n" * 8 + "<script>\n" + JS + "</script>\n"  # template-heavy AND embedded
    check = check_label(snip(text, "html", "a.html"))
    assert not check.ok and not check.hard
    assert "template-heavy" in check.reasons
    assert not check_label(snip(MARKUP + JS, "html", "a.js")).hard  # extension mismatch


def test_ok_snippet_is_not_hard():
    assert not check_label(snip(PY, "python", "a.py")).hard


def test_php_template_is_hard_php_code_is_not():
    template = "<?php get_header(); ?>\n" + MARKUP + "<?php get_footer(); ?>\n"
    check = check_label(snip(template, "php", "a.php"))
    assert check.hard and check.reasons == ("mostly embedded markup",)
    methods = "".join(f"  public function f{i}() {{ return {i}; }}\n" for i in range(12))
    code = "<?php\nclass A {\n" + methods + "}\n"
    assert check_label(snip(code, "php", "a.php")).ok
    some = code + "echo '<div>' . $x . '</div>';\n"  # PHP that prints a little HTML
    assert check_label(snip(some, "php", "a.php")).ok


def test_filter_labels_routes_hard_examples_when_asked():
    hard_text = "<p>a</p>\n<script>\n" + JS + "</script>\n"
    items = [snip(PY, "python", "a.py"), snip(hard_text, "html", "a.html")]
    kept, dropped = filter_labels(items)  # default: dropped like any other failure
    assert kept == [items[0]] and dropped["mostly embedded script/style"] == 1
    hard: list = []
    kept, dropped = filter_labels(items, hard)  # with a list: set aside, not dropped
    assert kept == [items[0]] and hard == [items[1]] and not dropped
