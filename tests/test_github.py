import json
import re
import time
from collections import Counter

import httpx
import pytest

from codelangtm import github as gh
from codelangtm.cli import main
from codelangtm.data import load_snippets

TOKEN = "test-token-do-not-leak"


def py_file(tag: str, n: int = 40) -> str:
    return "".join(
        f"def {tag}_{i}(value):\n    return value * {i} + len('{tag}')\n" for i in range(n)
    )


def repo_item(name, spdx="MIT", stars=60):
    return {
        "full_name": name,
        "license": {"spdx_id": spdx} if spdx else None,
        "default_branch": "main",
        "stargazers_count": stars,
    }


class FakeGitHub:
    """In-memory stand-in for api.github.com + raw.githubusercontent.com."""

    def __init__(self, repos, files, broken=()):
        self.repos = repos  # returned for the lowest star band only
        self.files = files  # {repo: {path: str | bytes}}
        self.broken = set(broken)  # repos whose commits endpoint returns 409 (empty repo)
        self.calls: Counter[str] = Counter()
        self.auth_headers: set[str] = set()

    def sha(self, repo):
        return "c0ffee" + repo.replace("/", "")

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.auth_headers.add(request.headers.get("Authorization", ""))
        path = request.url.path
        if request.url.host == "raw.githubusercontent.com":
            _, owner, name, _sha, rest = path.split("/", 4)
            self.calls[f"raw:{owner}/{name}"] += 1
            content = self.files[f"{owner}/{name}"][rest]
            body = content.encode() if isinstance(content, str) else content
            return httpx.Response(200, content=body)
        if path == "/search/repositories":
            self.calls["search"] += 1
            q = request.url.params["q"]
            return httpx.Response(200, json={"items": self.repos if "stars:50..199" in q else []})
        if m := re.fullmatch(r"/repos/([^/]+/[^/]+)/commits/(.+)", path):
            repo = m.group(1)
            self.calls[f"commits:{repo}"] += 1
            if repo in self.broken:
                return httpx.Response(409, json={"message": "Git Repository is empty."})
            return httpx.Response(200, json={"sha": self.sha(repo)})
        if m := re.fullmatch(r"/repos/([^/]+/[^/]+)/git/trees/(.+)", path):
            repo = m.group(1)
            self.calls[f"tree:{repo}"] += 1
            tree = [
                {"type": "blob", "path": p, "size": len(c)}
                for p, c in self.files.get(repo, {}).items()
            ]
            return httpx.Response(200, json={"tree": tree})
        return httpx.Response(404)


def make_fake():
    files = {
        repo: {f"src/m{i}.py": py_file(f"{repo.replace('/', '_')}_{i}") for i in range(6)}
        for repo in ("a/one", "a/two", "gpl/repo", "none/repo")
    }
    files["a/one"]["src/latin1.py"] = "x = 'caf\xe9'\n".encode("latin-1") * 50
    repos = [
        repo_item("a/one"),
        repo_item("gpl/repo", spdx="GPL-3.0"),
        repo_item("none/repo", spdx=None),
        repo_item("a/empty"),
        repo_item("a/two", spdx="Apache-2.0", stars=120),
    ]
    return FakeGitHub(repos, files, broken={"a/empty"})


def client_for(fake, tmp_path, sleeps=None):
    return gh.GitHubClient(
        token=TOKEN,
        cache_dir=tmp_path / "cache",
        transport=httpx.MockTransport(fake),
        sleep=(sleeps.append if sleeps is not None else lambda s: None),
    )


def test_star_buckets():
    assert gh.star_buckets(50) == ["50..199", "200..999", "1000..4999", ">=5000"]
    assert gh.star_buckets(3000) == ["3000..4999", ">=5000"]
    assert gh.star_buckets(10000) == [">=10000"]


def test_candidate_files_filters():
    def blob(path, size=1000):
        return {"type": "blob", "path": path, "size": size}

    tree = {
        "tree": [
            blob("src/ok.py"),
            blob("pkg/Mod.PY"),
            blob("vendor/lib.py"),
            blob("a/node_modules/b.py"),
            blob("build/gen.py"),
            blob("api_pb2.py"),
            blob("tiny.py", size=100),
            blob("huge.py", size=300_000),
            blob("app.js"),
            {"type": "tree", "path": "src"},
        ]
    }
    assert gh.candidate_files(tree, "python") == ["pkg/Mod.PY", "src/ok.py"]
    assert gh.candidate_files(tree, "javascript") == ["app.js"]


def test_collect_language_licenses_caps_and_fields(tmp_path):
    fake = make_fake()
    with client_for(fake, tmp_path) as client:
        snippets, report = gh.collect_language(client, "python", n_repos=5, per_repo=3)

    assert {s.repo for s in snippets} == {"a/one", "a/two"}
    assert all(c <= 3 for c in Counter(s.repo for s in snippets).values())
    assert fake.calls["commits:gpl/repo"] == 0  # disallowed licenses never touched
    assert fake.calls["commits:none/repo"] == 0
    assert report.repos_skipped == 1  # a/empty (409)
    assert report.snippets == len(snippets) == 6
    for s in snippets:
        assert s.source == "github"
        assert s.language == "python"
        assert s.commit == fake.sha(s.repo)
        assert s.license in {"MIT", "Apache-2.0"}
        assert s.text.count("\n") == s.end_line - s.start_line + 1
        assert s.text in fake.files[s.repo][s.path]
    manifest = {r["repo"]: r for r in report.to_dict()["repos"]}
    assert manifest["a/two"]["license"] == "Apache-2.0"
    assert manifest["a/one"]["commit"] == fake.sha("a/one")


def test_collect_is_deterministic(tmp_path):
    runs = []
    for i in range(2):
        with client_for(make_fake(), tmp_path / str(i)) as client:
            runs.append(gh.collect_language(client, "python", n_repos=2, per_repo=2)[0])
    assert runs[0] == runs[1]


def test_cache_means_no_redownload(tmp_path):
    fake = make_fake()
    with client_for(fake, tmp_path) as client:
        first = gh.collect_language(client, "python", n_repos=2, per_repo=2)[0]
    raw_calls = sum(v for k, v in fake.calls.items() if k.startswith("raw:"))
    tree_calls = sum(v for k, v in fake.calls.items() if k.startswith("tree:"))
    with client_for(fake, tmp_path) as client:
        second = gh.collect_language(client, "python", n_repos=2, per_repo=2)[0]
    assert second == first
    assert sum(v for k, v in fake.calls.items() if k.startswith("raw:")) == raw_calls
    assert sum(v for k, v in fake.calls.items() if k.startswith("tree:")) == tree_calls


def test_non_utf8_dropped(tmp_path):
    fake = make_fake()
    fake.files["a/one"] = {"src/latin1.py": "x = 'caf\xe9'\n".encode("latin-1") * 50}
    fake.repos = [repo_item("a/one")]
    with client_for(fake, tmp_path) as client:
        snippets, report = gh.collect_language(client, "python", n_repos=1, per_repo=1)
    assert snippets == []
    assert report.dropped["not utf-8"] == 1


def test_rate_limit_waits_then_retries(tmp_path):
    responses = [
        httpx.Response(
            403,
            headers={"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": str(int(time.time()) + 5)},
        ),
        httpx.Response(200, json={"ok": True}),
    ]
    sleeps: list[float] = []
    client = gh.GitHubClient(
        token=TOKEN, cache_dir=tmp_path, sleep=sleeps.append,
        transport=httpx.MockTransport(lambda req: responses.pop(0)),
    )  # fmt: skip
    assert client.get_json("/rate_limit") == {"ok": True}
    assert len(sleeps) == 1 and 1 <= sleeps[0] <= 7


def test_retry_after_and_server_errors(tmp_path):
    responses = [
        httpx.Response(429, headers={"Retry-After": "3"}),
        httpx.Response(502),
        httpx.Response(200, json={}),
    ]
    sleeps: list[float] = []
    client = gh.GitHubClient(
        token=TOKEN, cache_dir=tmp_path, sleep=sleeps.append,
        transport=httpx.MockTransport(lambda req: responses.pop(0)),
    )  # fmt: skip
    client.get_json("/x")
    assert sleeps == [3.0, 2.0]


def test_plain_403_fails_fast(tmp_path):
    sleeps: list[float] = []
    client = gh.GitHubClient(
        token=TOKEN, cache_dir=tmp_path, sleep=sleeps.append,
        transport=httpx.MockTransport(lambda req: httpx.Response(403)),
    )  # fmt: skip
    with pytest.raises(httpx.HTTPStatusError):
        client.get_json("/x")
    assert sleeps == []


def test_missing_token(monkeypatch, tmp_path):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="GITHUB_TOKEN"):
        gh.GitHubClient(cache_dir=tmp_path)


def test_token_sent_only_as_bearer_header(tmp_path):
    fake = make_fake()
    with client_for(fake, tmp_path) as client:
        gh.collect_language(client, "python", n_repos=1, per_repo=1)
    assert fake.auth_headers == {f"Bearer {TOKEN}"}


def test_cli_collect_end_to_end_without_leaking_token(tmp_path, monkeypatch, capsys):
    fake = make_fake()
    real_client = gh.GitHubClient
    monkeypatch.setattr(
        gh,
        "GitHubClient",
        lambda cache_dir: real_client(
            token=TOKEN, cache_dir=cache_dir, transport=httpx.MockTransport(fake),
            sleep=lambda s: None,
        ),
    )  # fmt: skip
    out = tmp_path / "out"
    code = main([
        "collect", "github", "--language", "python", "--repos", "2", "--per-repo", "2",
        "--out", str(out), "--cache", str(tmp_path / "cache"),
    ])  # fmt: skip
    assert code == 0

    snippets = list(load_snippets(out / "python.jsonl"))
    assert len(snippets) == 4
    manifest = json.loads((out / "python.manifest.json").read_text(encoding="utf-8"))
    assert manifest["snippets"] == 4 and manifest["params"]["per_repo"] == 2
    console = capsys.readouterr()
    assert "python: 4 snippets from 2 repos" in console.out
    assert TOKEN not in console.out + console.err

    for f in tmp_path.rglob("*"):
        if f.is_file():
            assert TOKEN.encode() not in f.read_bytes(), f"token leaked into {f}"


def test_cli_help_and_version_unchanged(capsys):
    assert main([]) == 0
    assert "collect" in capsys.readouterr().out
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0
