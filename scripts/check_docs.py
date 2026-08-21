"""Guard the repo's documentation coverage: required READMEs, plus a nudge
for new undocumented packages under rag_app/.

Unlike docstring/annotation coverage (see ruff.toml), "does this folder need
a README" isn't a rule ruff can express, and inferring it from directory
structure alone is unreliable: this project intentionally documents an
entire subsystem (e.g. rag_app/ingestion/parser/, chunking/, cleaners/,
embeddings/) in one parent README rather than one README per subpackage. A
generic "every package needs a README" rule would demand pointless READMEs
on those leaf folders.

Instead, REQUIRED_README_DIRS is an explicit allowlist of directories that
must carry a non-empty README.md. It fails the run (exit 1) if one goes
missing. When you intentionally add a new documented subsystem, write its
README and add its path here in the same change -- that's the one place to
update.

Anything else -- a new package under rag_app/ with real content that isn't
covered by an existing required README's documentation -- only gets a
warning printed to stderr, not a failure, so this never blocks a commit on a
false positive.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Directories that must contain a non-empty README.md. Add to this list when
# you intentionally create a new documented subsystem.
REQUIRED_README_DIRS = [
    PROJECT_ROOT,
    PROJECT_ROOT / "rag_app" / "ingestion",
    PROJECT_ROOT / "rag_app" / "vectorstore",
]

# Directories under rag_app/ whose contents are documented by an ancestor's
# README rather than one of their own, so the "undocumented package" scan
# below should not warn about them.
DOCUMENTED_BY_ANCESTOR = {
    PROJECT_ROOT / "rag_app" / "ingestion" / "parser",
    PROJECT_ROOT / "rag_app" / "ingestion" / "chunking",
    PROJECT_ROOT / "rag_app" / "ingestion" / "cleaners",
    PROJECT_ROOT / "rag_app" / "ingestion" / "embeddings",
}


def check_required_readmes() -> list[str]:
    """Return an error message per required directory missing a non-empty README.md."""

    errors = []
    for directory in REQUIRED_README_DIRS:
        readme = directory / "README.md"
        if not readme.is_file() or not readme.read_text(encoding="utf-8").strip():
            rel = (
                directory.relative_to(PROJECT_ROOT)
                if directory != PROJECT_ROOT
                else Path(".")
            )
            errors.append(f"Missing or empty README.md in {rel}")
    return errors


def warn_undocumented_packages() -> None:
    """Print a non-blocking warning for rag_app/ packages with content but no README."""

    rag_app = PROJECT_ROOT / "rag_app"
    required = {d for d in REQUIRED_README_DIRS if d != PROJECT_ROOT}

    for init_file in sorted(rag_app.rglob("__init__.py")):
        package_dir = init_file.parent
        if package_dir in required or package_dir in DOCUMENTED_BY_ANCESTOR:
            continue

        has_content = any(f.name != "__init__.py" for f in package_dir.glob("*.py"))
        if has_content and not (package_dir / "README.md").is_file():
            rel = package_dir.relative_to(PROJECT_ROOT)
            print(
                f"warning: {rel} has no README.md and isn't in "
                "REQUIRED_README_DIRS or DOCUMENTED_BY_ANCESTOR in "
                "scripts/check_docs.py -- add one or note where it's "
                "documented.",
                file=sys.stderr,
            )


def main() -> int:
    """Fail if a required README is missing; warn (without failing) about others."""

    errors = check_required_readmes()
    warn_undocumented_packages()

    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
