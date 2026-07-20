#!/usr/bin/env python3
"""
GitHub Portfolio Analyzer
==========================

Fetches every repository owned by a GitHub account, builds a compact
"skeleton" representation of each one (metadata + README + dependency
manifests + a depth-limited directory tree), and runs a 3-step local-LLM
prompt chain over it (tech-stack extraction -> maturity/debt audit ->
markdown writer) to produce `github_portfolio_report.md`.

Design notes (why this differs slightly from a minimal 3-step script):

- Deterministic front-matter: repo name, URL, stars, dates and the final
  maturity/complexity/confidence values are rendered by Python from real
  data, not written by the LLM. A 14B local model has no business being the
  source of truth for facts like star counts; it only ever writes the
  narrative prose. This also saves context budget in step 3, which no
  longer needs the raw README/tree at all - just the JSON from steps 1+2.
- Deterministic repo signals (has_tests, has_ci, has_lockfile, ...) are
  computed in Python from the tree and handed to the Auditor as facts
  instead of asking the model to eyeball a text tree for them - a smaller
  local model is much more reliable when it only has to *reason* over
  given facts rather than *extract* them from noisy text first.
- Per-repo caching: each repo's chapter is written to its own file under
  output/chapters/, keyed with a state.json entry storing the repo's
  `pushed_at` timestamp and a PIPELINE_VERSION. Re-running the script skips
  repos that haven't changed since their last successful analysis (and
  auto-invalidates the cache whenever the analysis pipeline itself changes),
  which matters a lot given how slow 3x LLM calls per repo add up across a
  whole account. The final report is reassembled from all chapter files
  after every repo, so it is always complete and up to date even if the
  run is interrupted.
- Real language composition: GitHub's `languages` endpoint (the exact byte
  counts behind the repo language bar in the GitHub UI) is used as ground
  truth for the "Languages" row instead of letting the LLM guess from
  prose, which used to produce "unknown" for repos with a thin README.
- Always-on source sampling: a diversified slice of real source files
  (entry points first, then files matching the repo's top languages,
  capped per directory) is always included in the skeleton, not just as a
  fallback for repos with a thin/missing README - so the LLM analyzes
  actual code, not just metadata and prose.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import ollama
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

# --------------------------------------------------------------------------
# 1. Constants & configuration
# --------------------------------------------------------------------------

GITHUB_API = "https://api.github.com"
REQUEST_TIMEOUT = 30
OLLAMA_NUM_CTX = 32768  # Must match the Modelfile's PARAMETER num_ctx.

# Directories/files that add noise, not signal, to a directory tree.
JUNK_DIR_NAMES = {
    ".git", "node_modules", "venv", ".venv", "env", "__pycache__", "dist",
    "build", ".next", "target", ".idea", ".vscode", "vendor", "bin", "obj",
    ".mypy_cache", ".pytest_cache", "coverage", ".gradle", ".tox", "out",
}

# Exact filenames recognized as dependency/package manifests.
DEPENDENCY_FILENAMES = {
    "package.json", "requirements.txt", "pyproject.toml", "Pipfile",
    "setup.py", "Cargo.toml", "go.mod", "pom.xml", "build.gradle",
    "build.gradle.kts", "composer.json", "Gemfile", "mix.exs",
    "environment.yml",
}
MAX_DEPENDENCY_FILES = 6

# Source-code sampling: always taken (not just when the README is thin), to
# give the LLM real code instead of just prose to analyze. Selection is
# diversified across entry points, top languages, and directories - see
# pick_source_sample_paths().
SOURCE_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java", ".rb",
    ".php", ".c", ".cpp", ".h", ".hpp", ".cs", ".kt", ".swift",
}
EXCLUDE_PATH_HINTS = ("test", "spec", "node_modules", "vendor", ".min.")
ENTRY_POINT_NAMES = {
    "main.py", "app.py", "manage.py", "index.js", "index.ts", "server.js",
    "main.go", "program.cs", "main.rs",
}
# Maps GitHub's `languages` API names to file extensions, used only to bias
# sampling toward the repo's actual top languages - not a hard filter.
LANGUAGE_TO_EXTENSIONS = {
    "Python": {".py"}, "JavaScript": {".js", ".jsx"}, "TypeScript": {".ts", ".tsx"},
    "Go": {".go"}, "Rust": {".rs"}, "Java": {".java"}, "Ruby": {".rb"},
    "PHP": {".php"}, "C": {".c", ".h"}, "C++": {".cpp", ".hpp"}, "C#": {".cs"},
    "Kotlin": {".kt"}, "Swift": {".swift"},
}
SOURCE_SAMPLE_MAX_FILES = 8
SOURCE_SAMPLE_MAX_CHARS = 4000
MAX_FILES_PER_DIR = 2  # diversification cap so one folder can't dominate the sample

# Context-budget caps for the skeleton (generous headroom under the 32k
# context: worst case is roughly 6k readme + 18k deps + 4k tree + 32k samples
# =~ 60k chars =~ 17k tokens, leaving ~10-12k tokens for instructions/output).
README_MAX_CHARS = 6000
DEP_FILE_MAX_CHARS = 3000
TREE_MAX_CHARS = 4000
TREE_MAX_DEPTH = 3

# Bumped whenever the analysis prompts/schema change, so cached chapters from
# an older pipeline version are automatically reprocessed instead of silently
# kept stale.
PIPELINE_VERSION = 2

# --------------------------------------------------------------------------
# 2. GitHub API client
# --------------------------------------------------------------------------


class GitHubClient:
    """Thin wrapper around the GitHub REST API used to build repo skeletons."""

    def __init__(self, token: str):
        self.session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=1.5,
            status_forcelist=(500, 502, 503, 504),
            allowed_methods=("GET",),
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retries))
        self.session.headers.update(
            {
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "github-portfolio-analyzer",
            }
        )
        if token:
            self.session.headers["Authorization"] = f"Bearer {token}"

    def _get(self, url: str, params: dict | None = None) -> requests.Response:
        while True:
            resp = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 403 and resp.headers.get("X-RateLimit-Remaining") == "0":
                reset = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
                wait = max(reset - time.time(), 1)
                logging.warning("GitHub rate limit exhausted, sleeping %.0fs", wait)
                time.sleep(wait + 1)
                continue
            return resp

    def get_authenticated_login(self) -> str | None:
        resp = self._get(f"{GITHUB_API}/user")
        if resp.status_code == 200:
            return resp.json().get("login")
        return None

    def list_user_repos(self, username: str, include_forks: bool) -> list[dict]:
        authed = "Authorization" in self.session.headers
        repos: list[dict] = []
        page = 1
        while True:
            if authed:
                # /user/repos returns the token owner's repos (incl. private
                # ones the token can see) - more useful than the public-only
                # /users/{username}/repos endpoint when a token is supplied.
                url = f"{GITHUB_API}/user/repos"
                params = {"per_page": 100, "page": page, "affiliation": "owner", "sort": "full_name"}
            else:
                url = f"{GITHUB_API}/users/{username}/repos"
                params = {"per_page": 100, "page": page, "sort": "full_name"}
            resp = self._get(url, params=params)
            resp.raise_for_status()
            batch = resp.json()
            if not batch:
                break
            repos.extend(batch)
            page += 1
        if not include_forks:
            repos = [r for r in repos if not r.get("fork")]
        return repos

    def get_readme(self, owner: str, repo: str) -> str | None:
        resp = self._get(f"{GITHUB_API}/repos/{owner}/{repo}/readme")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        if data.get("encoding") == "base64" and data.get("content"):
            return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
        return None

    def get_tree(self, owner: str, repo: str, branch: str | None) -> tuple[list[dict], bool]:
        """Returns (tree_entries, truncated). Empty list means empty/inaccessible repo."""
        if not branch:
            return [], True
        resp = self._get(f"{GITHUB_API}/repos/{owner}/{repo}/git/trees/{branch}", params={"recursive": 1})
        if resp.status_code in (404, 409):
            # 409 = empty repository (no commits yet on the default branch)
            return [], True
        resp.raise_for_status()
        data = resp.json()
        return data.get("tree", []), bool(data.get("truncated", False))

    def get_languages(self, owner: str, repo: str) -> dict[str, int]:
        """Bytes of code per language - the exact same data behind GitHub's UI language bar."""
        resp = self._get(f"{GITHUB_API}/repos/{owner}/{repo}/languages")
        if resp.status_code != 200:
            return {}
        return resp.json()

    def get_file_content(self, owner: str, repo: str, path: str, max_bytes: int = 200_000) -> str | None:
        resp = self._get(f"{GITHUB_API}/repos/{owner}/{repo}/contents/{path}")
        if resp.status_code != 200:
            return None
        data = resp.json()
        if isinstance(data, list):  # path was actually a directory
            return None
        if data.get("size", 0) > max_bytes:
            return None
        if data.get("encoding") == "base64" and data.get("content"):
            try:
                return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
            except (ValueError, UnicodeDecodeError):
                return None
        return None


# --------------------------------------------------------------------------
# 3. Skeleton construction (metadata + README + deps + tree, no source RAG)
# --------------------------------------------------------------------------


def format_language_breakdown(languages: dict[str, int]) -> str:
    """Renders GitHub's byte-based language stats as "Python 88.4%, HTML 7.1%, ..."."""
    total = sum(languages.values())
    if not total:
        return "not detected by GitHub"
    parts = sorted(languages.items(), key=lambda kv: kv[1], reverse=True)
    return ", ".join(f"{name} {100 * count / total:.1f}%" for name, count in parts)


def compute_repo_signals(tree_entries: list[dict]) -> dict:
    """Deterministic yes/no facts a human reviewer would look for first."""
    paths = [e["path"].lower() for e in tree_entries if e.get("type") == "blob"]

    def any_match(*needles: str) -> bool:
        return any(needle in p for needle in needles for p in paths)

    return {
        "has_tests": any_match("test/", "tests/", "spec/", "__tests__/", "test_", "_test."),
        "has_ci": any(p.startswith(".github/workflows/") for p in paths)
        or any_match(".gitlab-ci.yml", "azure-pipelines.yml", "jenkinsfile", ".circleci/"),
        "has_dockerfile": any_match("dockerfile", "docker-compose"),
        "has_license": any_match("license"),
        "has_lockfile": any_match(
            "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock",
            "cargo.lock", "gemfile.lock", "pipfile.lock",
        ),
        "file_count": len(paths),
    }


def build_tree_text(tree_entries: list[dict], max_depth: int = TREE_MAX_DEPTH) -> str:
    filtered = [
        e for e in tree_entries
        if not any(part in JUNK_DIR_NAMES for part in e["path"].split("/"))
        and e["path"].count("/") < max_depth
    ]
    filtered.sort(key=lambda e: e["path"])
    lines = []
    for e in filtered:
        depth = e["path"].count("/")
        name = e["path"].split("/")[-1]
        suffix = "/" if e["type"] == "tree" else ""
        lines.append(f"{'  ' * depth}{name}{suffix}")
    text = "\n".join(lines)
    if len(text) > TREE_MAX_CHARS:
        text = text[:TREE_MAX_CHARS] + "\n... (truncated)"
    return text or "(empty)"


def find_dependency_files(tree_entries: list[dict]) -> list[str]:
    matches = [
        e["path"] for e in tree_entries
        if e.get("type") == "blob" and e["path"].split("/")[-1] in DEPENDENCY_FILENAMES
    ]
    matches.sort(key=lambda p: (p.count("/"), p))  # prefer shallow/root files
    return matches[:MAX_DEPENDENCY_FILES]


def pick_source_sample_paths(tree_entries: list[dict], languages_bytes: dict[str, int]) -> list[str]:
    """Picks a diverse slice of source files: known entry points first, then files
    matching the repo's top languages, spread across directories rather than all
    coming from whichever single folder happens to hold the biggest files."""
    top_languages = [name for name, _ in sorted(languages_bytes.items(), key=lambda kv: kv[1], reverse=True)[:2]]
    preferred_extensions: set[str] = set()
    for lang in top_languages:
        preferred_extensions |= LANGUAGE_TO_EXTENSIONS.get(lang, set())

    candidates = []
    for e in tree_entries:
        if e.get("type") != "blob":
            continue
        path = e["path"]
        lower = path.lower()
        if os.path.splitext(path)[1] not in SOURCE_EXTENSIONS:
            continue
        if any(hint in lower for hint in EXCLUDE_PATH_HINTS):
            continue
        candidates.append(e)

    def sort_key(e: dict) -> tuple:
        basename = e["path"].split("/")[-1].lower()
        is_entry_point = basename not in ENTRY_POINT_NAMES
        matches_top_language = os.path.splitext(e["path"])[1] not in preferred_extensions
        return (is_entry_point, matches_top_language, -e.get("size", 0))

    candidates.sort(key=sort_key)

    selected: list[str] = []
    files_per_dir: dict[str, int] = {}
    for e in candidates:
        if len(selected) >= SOURCE_SAMPLE_MAX_FILES:
            break
        top_dir = e["path"].split("/")[0] if "/" in e["path"] else ""
        if files_per_dir.get(top_dir, 0) >= MAX_FILES_PER_DIR:
            continue
        selected.append(e["path"])
        files_per_dir[top_dir] = files_per_dir.get(top_dir, 0) + 1
    return selected


def build_skeleton(gh: GitHubClient, owner: str, name: str, repo_meta: dict) -> dict:
    tree_entries, truncated = gh.get_tree(owner, name, repo_meta.get("default_branch"))

    readme = (gh.get_readme(owner, name) or "").strip()
    languages_bytes = gh.get_languages(owner, name)

    dependency_files: dict[str, str] = {}
    for path in find_dependency_files(tree_entries):
        content = gh.get_file_content(owner, name, path)
        if content:
            dependency_files[path] = content[:DEP_FILE_MAX_CHARS]

    # Always sample source code (not just when the README is thin) so the LLM
    # has real code to analyze instead of relying on prose/metadata alone.
    sample_files: dict[str, str] = {}
    if tree_entries:
        for path in pick_source_sample_paths(tree_entries, languages_bytes):
            content = gh.get_file_content(owner, name, path)
            if content:
                sample_files[path] = content[:SOURCE_SAMPLE_MAX_CHARS]

    signals = compute_repo_signals(tree_entries)
    signals["readme_length"] = len(readme)
    signals["dependency_files_found"] = len(dependency_files)
    signals["tree_truncated_by_github"] = truncated

    return {
        "readme": readme[:README_MAX_CHARS],
        "dependency_files": dependency_files,
        "sample_files": sample_files,
        "tree_text": build_tree_text(tree_entries) if tree_entries else "(empty or inaccessible)",
        "signals": signals,
        "is_empty_repo": not tree_entries,
        "languages_bytes": languages_bytes,
        "languages_text": format_language_breakdown(languages_bytes),
    }


# --------------------------------------------------------------------------
# 4. Ollama prompt chain (step 1: tech parser, step 2: auditor, step 3: writer)
# --------------------------------------------------------------------------

TECH_PARSER_SYSTEM = (
    "You are a meticulous technology-stack analyst. You are given partial, "
    "possibly incomplete evidence about a code repository (README excerpt, "
    "dependency manifests, a directory tree). Identify only what the "
    "evidence actually supports - never invent library or framework names. "
    "Respond with ONLY a single valid JSON object, no markdown fences, no "
    "commentary before or after it."
)

AUDITOR_SYSTEM = (
    "You are a pragmatic engineering auditor reviewing a personal/portfolio "
    "repository. Many such repositories are unfinished side projects, so do "
    "not assume best practices were followed - judge strictly from the "
    "signals given. Respond with ONLY a single valid JSON object, no "
    "markdown fences, no commentary before or after it."
)

WRITER_SYSTEM = (
    "You are a technical writer turning structured repository analysis data "
    "into the narrative body of one report chapter. Be thorough and factual, "
    "using the full detail given to you rather than compressing it into "
    "generic filler; do not add information beyond what is given. Output "
    "clean Markdown only, starting directly with a '### Summary' heading - "
    "no repository title, no metadata table, and no surrounding code fences."
)


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def call_ollama_json(
    client: ollama.Client,
    model: str,
    system_prompt: str,
    user_prompt: str,
    required_keys: list[str],
    defaults: dict,
    num_predict: int | None = None,
) -> dict:
    """Calls Ollama with format="json" and retries once on malformed output
    before falling back to safe defaults, so one bad repo can't crash the run."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    options = {"num_ctx": OLLAMA_NUM_CTX, "temperature": 0.2}
    if num_predict is not None:
        options["num_predict"] = num_predict
    for attempt in range(2):
        try:
            response = client.chat(
                model=model,
                messages=messages,
                format="json",
                options=options,
            )
            content = response["message"]["content"]
        except Exception as exc:  # local server hiccup, timeout, etc.
            logging.error("Ollama call failed (attempt %d): %s", attempt + 1, exc)
            time.sleep(2)
            continue

        parsed = _extract_json(content)
        if parsed and all(key in parsed for key in required_keys):
            return parsed

        logging.warning("Invalid JSON from model on attempt %d, retrying once...", attempt + 1)
        messages.append({"role": "assistant", "content": content})
        messages.append(
            {
                "role": "user",
                "content": (
                    "Your previous reply was not valid JSON with exactly the required keys. "
                    "Reply again with ONLY a single valid JSON object containing exactly these "
                    "keys: " + ", ".join(required_keys)
                ),
            }
        )

    logging.error("Falling back to default values after repeated invalid JSON output.")
    return dict(defaults)


def step1_tech_parser(client: ollama.Client, model: str, repo_meta: dict, skeleton: dict) -> dict:
    dep_section = "\n\n".join(
        f"--- {path} ---\n{content}" for path, content in skeleton["dependency_files"].items()
    ) or "(none found)"
    sample_section = "\n\n".join(
        f"--- {path} ---\n{content}" for path, content in skeleton["sample_files"].items()
    )

    user_prompt = f"""Repository: {repo_meta['full_name']}
Real language composition (ground truth from GitHub, by bytes of code - do not contradict this):
{skeleton['languages_text']}

Directory tree (depth <= {TREE_MAX_DEPTH}, common build/dependency folders filtered out):
{skeleton['tree_text']}

Dependency / manifest files found:
{dep_section}

README (may be truncated, or absent):
{skeleton['readme'] or '(no README found)'}
{("Sampled source files (a representative slice of the actual code, not the whole repo):\n" + sample_section) if sample_section else ""}

Task: Based ONLY on the evidence above, identify the technology stack and how the code is organized.
Respond with a single JSON object with exactly these keys:
- "languages": array of programming languages actually evidenced - must be consistent with the real language composition given above
- "frameworks": array of frameworks/major libraries evidenced (empty array if none)
- "libraries": array of other notable libraries/tools evidenced
- "architecture_patterns": array of architecture patterns evidenced (e.g. "REST API", "CLI tool", "monorepo") - empty array if not evidenced
- "project_type": one short string describing what kind of project this is
- "architecture_notes": 2-4 sentences on how the code appears to be organized (module boundaries, entry points, separation of concerns) based on the tree and sampled files - say explicitly if the sample was too small to tell
- "notes": short string with anything notable or ambiguous (empty string if nothing)
If the evidence is too sparse to tell, use empty arrays and say so in "notes" - do not guess.
"""
    defaults = {
        "languages": [name for name, _ in sorted(skeleton["languages_bytes"].items(), key=lambda kv: kv[1], reverse=True)]
        or ([repo_meta["language"]] if repo_meta.get("language") else []),
        "frameworks": [],
        "libraries": [],
        "architecture_patterns": [],
        "project_type": "unknown",
        "architecture_notes": "LLM analysis failed; no architecture notes available.",
        "notes": "LLM analysis failed; falling back to GitHub-reported language data only.",
    }
    return call_ollama_json(client, model, TECH_PARSER_SYSTEM, user_prompt, list(defaults.keys()), defaults)


def step2_auditor(client: ollama.Client, model: str, repo_meta: dict, skeleton: dict, tech_json: dict) -> dict:
    signals = skeleton["signals"]
    user_prompt = f"""Repository: {repo_meta['full_name']}
Archived on GitHub: {repo_meta.get('archived')}
Last pushed: {repo_meta.get('pushed_at')}
Stars: {repo_meta.get('stargazers_count')}
Open issues: {repo_meta.get('open_issues_count')}

Detected tech stack (from a previous analysis step):
{json.dumps(tech_json, ensure_ascii=False)}

Deterministic repository signals:
- Has a tests directory/files: {signals['has_tests']}
- Has CI configuration: {signals['has_ci']}
- Has a Dockerfile: {signals['has_dockerfile']}
- Has a LICENSE file: {signals['has_license']}
- Has a dependency lockfile: {signals['has_lockfile']}
- README length (characters): {signals['readme_length']}
- Dependency manifest files found: {signals['dependency_files_found']}
- Tracked files seen (depth-limited, may undercount): {signals['file_count']}
- Repository appears empty/inaccessible: {skeleton['is_empty_repo']}

Task: Assess this repository's maturity and technical health based ONLY on the evidence above.
Respond with a single JSON object with exactly these keys:
- "maturity": one of "PoC", "MVP", "Production", "Archived"
- "complexity": one of "Low", "Medium", "High"
- "technical_debt": array of full sentences, each stating a concrete debt item AND why it matters for
  this specific repo (empty array if none evidenced) - do not write bare fragments like "No tests"
- "strengths": array of full sentences (empty array if none evidenced)
- "risks": array of full sentences (empty array if none evidenced)
- "recommendations": array of concrete, actionable next steps for this repo specifically, ordered by
  impact (empty array if the repo is already in good shape)
- "confidence": one of "Low", "Medium", "High" - your confidence given how much evidence was available
If the repository looks empty or the evidence is too sparse, set "maturity" to "PoC", "confidence" to "Low", and explain why in "technical_debt".
"""
    defaults = {
        "maturity": "Archived" if repo_meta.get("archived") else "PoC",
        "complexity": "Low",
        "technical_debt": ["LLM analysis failed; this is a placeholder assessment."],
        "strengths": [],
        "risks": [],
        "recommendations": [],
        "confidence": "Low",
    }
    return call_ollama_json(
        client, model, AUDITOR_SYSTEM, user_prompt, list(defaults.keys()), defaults, num_predict=700
    )


def render_fallback_narrative(tech_json: dict, audit_json: dict) -> str:
    """Guarantees a chapter body even if the writer step itself fails outright."""
    stack = ", ".join(tech_json.get("languages", []) + tech_json.get("frameworks", [])) or "unknown stack"
    lines = [
        "### Summary",
        "",
        f"_Automated writer step failed; showing raw analysis._ Detected stack: {stack}.",
    ]
    if tech_json.get("architecture_notes"):
        lines += ["", "### Architecture & Code Organization", "", tech_json["architecture_notes"]]
    lines += [
        "",
        "### Assessment",
        f"Maturity: **{audit_json.get('maturity', 'unknown')}**, "
        f"Complexity: **{audit_json.get('complexity', 'unknown')}**, "
        f"Confidence: **{audit_json.get('confidence', 'unknown')}**",
    ]
    if audit_json.get("strengths"):
        lines += ["", "**Strengths**"] + [f"- {item}" for item in audit_json["strengths"]]
    debt_and_risks = audit_json.get("technical_debt", []) + audit_json.get("risks", [])
    if debt_and_risks:
        lines += ["", "**Technical Debt & Risks**"] + [f"- {item}" for item in debt_and_risks]
    if audit_json.get("recommendations"):
        lines += ["", "**Recommendations**"] + [f"- {item}" for item in audit_json["recommendations"]]
    return "\n".join(lines)


def step3_writer(client: ollama.Client, model: str, repo_meta: dict, tech_json: dict, audit_json: dict) -> str:
    user_prompt = f"""Repository: {repo_meta['full_name']}
Description (from GitHub, may be empty): {repo_meta.get('description') or '(none)'}

Tech stack analysis (JSON):
{json.dumps(tech_json, ensure_ascii=False, indent=2)}

Maturity/quality assessment (JSON):
{json.dumps(audit_json, ensure_ascii=False, indent=2)}

Write the narrative body of this repository's report chapter in clean Markdown, using
exactly these three subsections and nothing else:

### Summary
5-8 sentences covering: what the project actually does, its real tech stack (languages,
frameworks, libraries), and its project type. Be specific and use the detail you were given
instead of compressing it into generic filler.

### Architecture & Code Organization
2-4 sentences based on "architecture_notes" - how the code is organized, entry points,
module boundaries. If architecture_notes says the evidence was too thin, say so briefly
instead of inventing structure.

### Assessment
A paragraph on maturity, complexity and confidence, followed by bullet lists titled
"Strengths", "Technical Debt & Risks" (merge technical_debt and risks) and
"Recommendations" (from the recommendations array). Omit any list entirely if its
source array was empty rather than inventing filler content.
"""
    try:
        response = client.chat(
            model=model,
            messages=[{"role": "system", "content": WRITER_SYSTEM}, {"role": "user", "content": user_prompt}],
            options={"num_ctx": OLLAMA_NUM_CTX, "temperature": 0.3, "num_predict": 900},
        )
        text = response["message"]["content"].strip()
        if text:
            return text
    except Exception as exc:
        logging.error("Writer step failed for %s: %s", repo_meta["full_name"], exc)
    return render_fallback_narrative(tech_json, audit_json)


# --------------------------------------------------------------------------
# 5. Chapter rendering & report assembly
# --------------------------------------------------------------------------


def safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def render_chapter(repo_meta: dict, tech_json: dict, audit_json: dict, narrative_body: str, languages_text: str) -> str:
    frameworks = ", ".join(tech_json.get("frameworks", [])) or "-"
    table = (
        f"## {repo_meta['name']}\n\n"
        "| | |\n"
        "|---|---|\n"
        f"| **URL** | {repo_meta['html_url']} |\n"
        f"| **Description** | {repo_meta.get('description') or '-'} |\n"
        f"| **Languages** | {languages_text} |\n"
        f"| **Frameworks / Libraries** | {frameworks} |\n"
        f"| **Stars** | {repo_meta.get('stargazers_count', 0)} |\n"
        f"| **Last Push** | {repo_meta.get('pushed_at', 'unknown')} |\n"
        f"| **Archived** | {'yes' if repo_meta.get('archived') else 'no'} |\n"
        f"| **Maturity** | {audit_json.get('maturity', 'unknown')} |\n"
        f"| **Complexity** | {audit_json.get('complexity', 'unknown')} |\n"
        f"| **Assessment Confidence** | {audit_json.get('confidence', 'unknown')} |\n"
    )
    return table + "\n" + narrative_body.strip() + "\n"


def render_error_chapter(repo_meta: dict, reason: str) -> str:
    return (
        f"## {repo_meta['name']}\n\n"
        "| | |\n|---|---|\n"
        f"| **URL** | {repo_meta['html_url']} |\n"
        "| **Status** | Analysis failed |\n\n"
        f"_This repository could not be analyzed: {reason}._\n"
    )


def rebuild_report(chapters_dir: Path, repos: list[dict], output_path: Path, username: str) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# GitHub Portfolio Report - {username}",
        "",
        f"_Generated {now} - {len(repos)} repositories._",
        "",
        "## Table of Contents",
        "",
    ]
    for repo in repos:
        anchor = re.sub(r"[^a-z0-9-]", "", repo["name"].lower().replace(" ", "-"))
        lines.append(f"- [{repo['name']}](#{anchor})")
    lines += ["", "---", ""]

    for repo in repos:
        chapter_path = chapters_dir / f"{safe_filename(repo['full_name'])}.md"
        if chapter_path.exists():
            lines.append(chapter_path.read_text(encoding="utf-8"))
            lines.append("\n---\n")

    output_path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------
# 6. State / cache management
# --------------------------------------------------------------------------


def load_state(state_path: Path) -> dict:
    if state_path.exists():
        try:
            return json.loads(state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logging.warning("State file unreadable, starting fresh.")
    return {}


def save_state(state_path: Path, state: dict) -> None:
    state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------
# 7. Per-repo pipeline
# --------------------------------------------------------------------------


def process_repo(
    gh: GitHubClient,
    client: ollama.Client,
    model: str,
    repo_meta: dict,
    chapters_dir: Path,
    state: dict,
    force: bool,
) -> str:
    full_name = repo_meta["full_name"]
    chapter_path = chapters_dir / f"{safe_filename(full_name)}.md"
    cached = state.get(full_name)
    if (
        not force
        and cached
        and cached.get("status") == "done"
        and cached.get("pushed_at") == repo_meta.get("pushed_at")
        and cached.get("pipeline_version") == PIPELINE_VERSION
        and chapter_path.exists()
    ):
        return "cached"

    owner, name = full_name.split("/", 1)
    now_iso = lambda: datetime.now(timezone.utc).isoformat()

    try:
        skeleton = build_skeleton(gh, owner, name, repo_meta)
    except Exception as exc:
        logging.exception("Failed to fetch GitHub data for %s", full_name)
        chapter_path.write_text(render_error_chapter(repo_meta, f"GitHub data collection failed ({exc})"), encoding="utf-8")
        state[full_name] = {"status": "error", "pushed_at": repo_meta.get("pushed_at"), "analyzed_at": now_iso()}
        return "error"

    if skeleton["is_empty_repo"] and not skeleton["readme"]:
        chapter_path.write_text(
            render_error_chapter(repo_meta, "repository is empty (no commits / no default branch)"), encoding="utf-8"
        )
        state[full_name] = {
            "status": "done",
            "pushed_at": repo_meta.get("pushed_at"),
            "analyzed_at": now_iso(),
            "pipeline_version": PIPELINE_VERSION,
        }
        return "empty"

    try:
        tech_json = step1_tech_parser(client, model, repo_meta, skeleton)
        audit_json = step2_auditor(client, model, repo_meta, skeleton, tech_json)
        narrative = step3_writer(client, model, repo_meta, tech_json, audit_json)
        chapter = render_chapter(repo_meta, tech_json, audit_json, narrative, skeleton["languages_text"])
        chapter_path.write_text(chapter, encoding="utf-8")
        state[full_name] = {
            "status": "done",
            "pushed_at": repo_meta.get("pushed_at"),
            "analyzed_at": now_iso(),
            "pipeline_version": PIPELINE_VERSION,
        }
        return "done"
    except Exception as exc:
        logging.exception("LLM analysis failed for %s", full_name)
        chapter_path.write_text(render_error_chapter(repo_meta, f"LLM analysis failed ({exc})"), encoding="utf-8")
        state[full_name] = {"status": "error", "pushed_at": repo_meta.get("pushed_at"), "analyzed_at": now_iso()}
        return "error"


# --------------------------------------------------------------------------
# 8. Ollama connectivity check
# --------------------------------------------------------------------------


def get_available_models(client: ollama.Client) -> set[str]:
    """Handles both dict-style and pydantic-object responses across ollama-python versions."""
    result = client.list()
    models_field = result.get("models") if isinstance(result, dict) else getattr(result, "models", [])
    names = set()
    for m in models_field:
        name = m.get("model") if isinstance(m, dict) else getattr(m, "model", None)
        if name:
            names.add(name)
    return names


# --------------------------------------------------------------------------
# 9. Entry point
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze every GitHub repository of a user with a local LLM and produce a Markdown portfolio report."
    )
    parser.add_argument("--model", default="repo-analyzer-qwen", help="Ollama model name (built from Modelfile).")
    parser.add_argument("--output-dir", default="output", help="Directory for chapters and the state/cache file.")
    parser.add_argument("--report", default="github_portfolio_report.md", help="Path of the final assembled report.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N repos - useful for a quick test run.")
    parser.add_argument("--include-forks", action="store_true", help="Include forked repositories (excluded by default).")
    parser.add_argument("--force", action="store_true", help="Ignore the cache and re-analyze every repository.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("analyzer.log", encoding="utf-8")],
    )

    load_dotenv()
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    username = os.environ.get("GITHUB_USERNAME", "").strip()

    if not username or username == "replace_with_your_github_username":
        logging.error("GITHUB_USERNAME is not set in .env")
        sys.exit(1)
    if not token or token == "ghp_replace_with_your_own_token":
        logging.warning("GITHUB_TOKEN is not set - continuing unauthenticated (60 req/hour, public repos only).")
        token = ""

    output_dir = Path(args.output_dir)
    chapters_dir = output_dir / "chapters"
    chapters_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / "state.json"
    report_path = Path(args.report)

    client = ollama.Client()
    try:
        available_models = get_available_models(client)
    except Exception as exc:
        logging.error("Could not reach the Ollama server at http://localhost:11434 (%s). Is 'ollama serve' running?", exc)
        sys.exit(1)
    if args.model not in available_models and not any(m.startswith(args.model + ":") for m in available_models):
        logging.error("Model '%s' not found in Ollama. Build it first: ollama create %s -f Modelfile", args.model, args.model)
        sys.exit(1)

    gh = GitHubClient(token)

    if token:
        actual_login = gh.get_authenticated_login()
        if actual_login and actual_login.lower() != username.lower():
            logging.warning("GITHUB_TOKEN belongs to '%s', not GITHUB_USERNAME='%s'. Using the token's account.", actual_login, username)
            username = actual_login

    logging.info("Fetching repository list for %s...", username)
    repos = gh.list_user_repos(username, include_forks=args.include_forks)
    repos.sort(key=lambda r: r["full_name"].lower())
    if args.limit:
        repos = repos[: args.limit]
    logging.info("%d repositories to process.", len(repos))

    state = load_state(state_path)
    counts: dict[str, int] = {}

    for repo_meta in tqdm(repos, desc="Analyzing repositories", unit="repo"):
        result = process_repo(gh, client, args.model, repo_meta, chapters_dir, state, args.force)
        counts[result] = counts.get(result, 0) + 1
        # Persist after every repo so an interrupted run loses nothing.
        save_state(state_path, state)
        rebuild_report(chapters_dir, repos, report_path, username)

    logging.info("Done. %s", counts)
    logging.info("Report written to %s", report_path.resolve())


if __name__ == "__main__":
    main()
