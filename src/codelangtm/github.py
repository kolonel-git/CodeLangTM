"""GitHub collector: permissively licensed repos -> labelled Snippet records.

Auth: the token is read from GITHUB_TOKEN and only ever sent in the Authorization header.
It is never logged, cached or written to outputs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from itertools import zip_longest
from pathlib import Path, PurePosixPath
from urllib.parse import quote

import httpx

from .data import Snippet
from .dedup import dedup
from .labels import EXTENSION_LANGUAGE, check_label
from .windows import extract_windows

log = logging.getLogger(__name__)

API = "https://api.github.com"
RAW = "https://raw.githubusercontent.com"

ALLOWED_LICENSES = frozenset({"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause"})

# Our label -> GitHub search `language:` qualifier.
SEARCH_LANGUAGE = {
    "python": "Python",
    "cpp": "C++",
    "java": "Java",
    "javascript": "JavaScript",
    "rust": "Rust",
    "go": "Go",
    "sql": "SQL",
    "html": "HTML",
}

SKIP_DIRS = frozenset({
    "vendor", "vendored", "node_modules", "third_party", "third-party", "thirdparty",
    "external", "deps", "dist", "build", "target", "out", "site-packages",
    "__pycache__", "generated", "gen", ".git",
})  # fmt: skip
SKIP_SUFFIXES = (".min.js", ".min.css", ".pb.go", "_pb2.py", ".generated.cs", ".d.ts")
MIN_FILE_BYTES = 400
MAX_FILE_BYTES = 200_000
FILES_PER_SNIPPET = 3  # fetch budget per repo = per_repo * this


@dataclass(frozen=True)
class Repo:
    full_name: str
    license: str  # SPDX id as reported by GitHub
    default_branch: str
    stars: int


@dataclass
class CollectReport:
    language: str
    repos: list[dict] = field(default_factory=list)
    repos_skipped: int = 0
    files_fetched: int = 0
    dropped: Counter[str] = field(default_factory=Counter)
    duplicates_removed: int = 0
    snippets: int = 0

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "snippets": self.snippets,
            "repos": self.repos,
            "repos_skipped": self.repos_skipped,
            "files_fetched": self.files_fetched,
            "duplicates_removed": self.duplicates_removed,
            "dropped": dict(self.dropped.most_common()),
        }

    def summary(self) -> str:
        top = ", ".join(f"{k}: {v}" for k, v in self.dropped.most_common(4)) or "none"
        return (
            f"{self.language}: {self.snippets} snippets from {len(self.repos)} repos "
            f"({self.repos_skipped} repos skipped, {self.files_fetched} files fetched, "
            f"{self.duplicates_removed} duplicates removed; dropped: {top})"
        )


def _retry_wait(r: httpx.Response, attempt: int) -> float | None:
    """Seconds to wait before retrying, or None if the response should not be retried."""
    if r.status_code in (403, 429):
        if "Retry-After" in r.headers:
            return float(r.headers["Retry-After"])
        if r.headers.get("X-RateLimit-Remaining") == "0":
            reset = float(r.headers.get("X-RateLimit-Reset", "0"))
            return max(reset - time.time(), 0.0) + 1.0
        return None  # plain 403 = permission problem; retrying will not help
    if r.status_code >= 500:
        return 2.0**attempt
    return None


class GitHubClient:
    def __init__(
        self,
        token: str | None = None,
        cache_dir: str | Path = "data/cache/github",
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        max_retries: int = 3,
        max_wait: float = 3600.0,
    ) -> None:
        token = token or os.environ.get("GITHUB_TOKEN")
        if not token:
            raise RuntimeError(
                "GITHUB_TOKEN is not set. Create a read-only fine-grained token and set it as "
                "an environment variable (see docs/data-sources.md)."
            )
        self.cache_dir = Path(cache_dir)
        self.sleep = sleep
        self.max_retries = max_retries
        self.max_wait = max_wait
        self._http = httpx.Client(
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "codelangtm-collector",
            },
            transport=transport,
            timeout=30.0,
            follow_redirects=True,
        )

    def __enter__(self) -> GitHubClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    def _get(self, url: str, params: dict | None = None) -> httpx.Response:
        for attempt in range(self.max_retries + 1):
            r = self._http.get(url, params=params)
            wait = _retry_wait(r, attempt)
            if wait is None or attempt == self.max_retries:
                r.raise_for_status()
                return r
            if wait > self.max_wait:
                raise RuntimeError(f"GitHub rate limit resets in {wait:.0f}s (> max_wait)")
            log.warning("GitHub returned %s; retrying in %.0fs", r.status_code, wait)
            self.sleep(wait)
        raise AssertionError("unreachable")

    def get_json(self, path: str, params: dict | None = None) -> dict:
        return self._get(API + path, params).json()

    def _repo_cache(self, repo: str, sha: str) -> Path:
        return self.cache_dir / repo.replace("/", "__") / sha

    def get_tree(self, repo: str, sha: str) -> dict:
        """Recursive file tree at a commit (cached: a commit's tree never changes)."""
        cached = self._repo_cache(repo, sha) / "tree.json"
        if cached.exists():
            return json.loads(cached.read_text(encoding="utf-8"))
        data = self.get_json(f"/repos/{repo}/git/trees/{sha}", {"recursive": "1"})
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps(data), encoding="utf-8")
        return data

    def fetch_file(self, repo: str, sha: str, path: str) -> str | None:
        """File contents at a commit (cached); None if not valid UTF-8."""
        cached = self._repo_cache(repo, sha) / (hashlib.sha1(path.encode()).hexdigest() + ".src")
        if cached.exists():
            raw = cached.read_bytes()
        else:
            raw = self._get(f"{RAW}/{repo}/{sha}/{quote(path)}").content
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_bytes(raw)
        try:
            return raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            return None


def star_buckets(min_stars: int) -> list[str]:
    """Popularity bands so the dataset is not only famous-repo style."""
    edges = sorted({min_stars, *(e for e in (200, 1000, 5000) if e > min_stars)})
    bands = [f"{lo}..{hi - 1}" for lo, hi in zip(edges, edges[1:], strict=False)]
    return [*bands, f">={edges[-1]}"]


def search_repos(
    client: GitHubClient,
    language: str,
    limit: int,
    min_stars: int = 50,
    search_delay: float = 2.0,
) -> list[Repo]:
    """Permissively licensed, non-fork, non-archived repos, round-robin across star bands."""
    qualifier = SEARCH_LANGUAGE[language]
    per_band: list[list[Repo]] = []
    for band in star_buckets(min_stars):
        q = f'language:"{qualifier}" stars:{band} fork:false archived:false'
        data = client.get_json("/search/repositories", {"q": q, "per_page": 100})
        client.sleep(search_delay)  # search API allows 30 requests/minute
        repos = []
        for item in data.get("items", []):
            spdx = (item.get("license") or {}).get("spdx_id") or ""
            if spdx.lower() in ALLOWED_LICENSES:
                repos.append(
                    Repo(item["full_name"], spdx, item["default_branch"], item["stargazers_count"])
                )
        per_band.append(repos)

    out: list[Repo] = []
    seen: set[str] = set()
    for group in zip_longest(*per_band):
        for repo in group:
            if repo is not None and repo.full_name not in seen:
                seen.add(repo.full_name)
                out.append(repo)
                if len(out) == limit:
                    return out
    return out


def candidate_files(tree: dict, language: str) -> list[str]:
    """Paths worth sampling: right extension, sane size, not vendored/generated/minified."""
    exts = {ext for ext, lang in EXTENSION_LANGUAGE.items() if lang == language}
    out = []
    for entry in tree.get("tree", []):
        if entry.get("type") != "blob":
            continue
        p = PurePosixPath(entry["path"])
        if p.suffix.lower() not in exts:
            continue
        if not MIN_FILE_BYTES <= entry.get("size", 0) <= MAX_FILE_BYTES:
            continue
        if p.name.lower().endswith(SKIP_SUFFIXES):
            continue
        if any(part.lower() in SKIP_DIRS for part in p.parts[:-1]):
            continue
        out.append(entry["path"])
    return sorted(out)


def collect_repo(
    client: GitHubClient,
    repo: Repo,
    language: str,
    per_repo: int,
    rng: random.Random,
    report: CollectReport,
) -> tuple[str, list[Snippet]]:
    """Pin HEAD, sample files, take one random window per file. Returns (commit sha, snippets)."""
    branch = quote(repo.default_branch, safe="")
    sha = client.get_json(f"/repos/{repo.full_name}/commits/{branch}")["sha"]
    files = candidate_files(client.get_tree(repo.full_name, sha), language)
    rng.shuffle(files)

    out: list[Snippet] = []
    for path in files[: per_repo * FILES_PER_SNIPPET]:
        text = client.fetch_file(repo.full_name, sha, path)
        report.files_fetched += 1
        if text is None:
            report.dropped["not utf-8"] += 1
            continue
        windows = extract_windows(text, seed=rng.randrange(2**31), language=language)
        if not windows:
            report.dropped["no usable window"] += 1
            continue
        w = rng.choice(windows)
        snippet = Snippet(
            w.text, language, "github", repo.full_name, sha, path, repo.license,
            w.start_line, w.end_line,
        )  # fmt: skip
        check = check_label(snippet)
        if not check.ok:
            report.dropped.update(check.reasons)
            continue
        out.append(snippet)
        if len(out) >= per_repo:
            break
    return sha, out


def collect_language(
    client: GitHubClient,
    language: str,
    n_repos: int = 25,
    per_repo: int = 5,
    min_stars: int = 50,
    seed: int = 0,
    search_delay: float = 2.0,
) -> tuple[list[Snippet], CollectReport]:
    if language not in SEARCH_LANGUAGE:
        raise ValueError(f"unsupported language: {language!r}")
    rng = random.Random(f"{seed}:{language}")
    report = CollectReport(language)
    snippets: list[Snippet] = []
    commits: dict[str, tuple[str, Repo]] = {}

    for repo in search_repos(client, language, n_repos * 3, min_stars, search_delay):
        if len(commits) >= n_repos:
            break
        try:
            sha, got = collect_repo(client, repo, language, per_repo, rng, report)
        except httpx.HTTPStatusError as e:  # deleted, empty (409) or blocked repos
            log.warning("skipping %s: HTTP %s", repo.full_name, e.response.status_code)
            report.repos_skipped += 1
            continue
        if not got:
            report.repos_skipped += 1
            continue
        commits[repo.full_name] = (sha, repo)
        snippets.extend(got)

    kept, stats = dedup(snippets)
    report.duplicates_removed = stats.exact_removed + stats.near_removed
    report.snippets = len(kept)
    per_repo_kept = Counter(s.repo for s in kept)
    report.repos = [
        {
            "repo": name,
            "commit": sha,
            "license": repo.license,
            "stars": repo.stars,
            "snippets": per_repo_kept[name],
        }
        for name, (sha, repo) in commits.items()
    ]
    return kept, report
