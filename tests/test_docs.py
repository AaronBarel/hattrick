"""Keep docs/ honest about what the code actually contains.

Documentation drift is not cosmetic in this project. Two of the worst bugs
found here were introduced or prolonged by confident, wrong documentation: odds
columns labeled "closing" that were collected days before kickoff, and a
project layout describing a package that never existed. A doc nobody can trust
is worse than no doc, because it stops people from checking.

These are structural checks only - nothing here can verify that prose is
*true*, just that it has not fallen out of sync with the file tree.
"""
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"
SRC = REPO / "src" / "hattrick"

EXPECTED_DOCS = [
    "README.md", "architecture.md", "data.md", "modeling.md",
    "evaluation.md", "findings.md", "assumptions.md",
    "references.md",
]


def _modules():
    """Every importable module, as dotted paths relative to the package."""
    out = []
    for p in sorted(SRC.rglob("*.py")):
        if p.name == "__init__.py" or "__pycache__" in p.parts:
            continue
        out.append(p.relative_to(SRC).with_suffix("").as_posix())
    return out


def test_docs_directory_exists():
    assert DOCS.is_dir(), "docs/ is missing"


@pytest.mark.parametrize("name", EXPECTED_DOCS)
def test_expected_doc_present_and_not_empty(name):
    p = DOCS / name
    assert p.exists(), f"docs/{name} is missing"
    assert len(p.read_text().strip()) > 200, f"docs/{name} looks like a stub"


def test_every_module_appears_in_the_architecture_map():
    """Adding a module without documenting it fails the suite.

    This is the one piece of the update policy that is enforced rather than
    trusted, because an architecture map that silently omits modules is
    exactly how a reader ends up with a wrong mental model.
    """
    text = (DOCS / "architecture.md").read_text()
    missing = [m for m in _modules() if Path(m).name not in text]
    assert not missing, (
        f"modules absent from docs/architecture.md: {missing}. "
        f"Add them to the Modules section, or note them as removed."
    )


def test_architecture_map_does_not_describe_modules_that_are_gone():
    """The reverse drift: a doc still describing a deleted module."""
    text = (DOCS / "architecture.md").read_text()
    real = {Path(m).name + ".py" for m in _modules()}
    # Compare basenames: the doc legitimately writes some paths with their
    # directory ("data/xg.py"), which is the same file as "xg.py".
    referenced = {
        Path(w.strip("`*,.()")).name for w in text.replace("\n", " ").split()
        if w.strip("`*,.()").endswith(".py")
    }
    known_scripts = {p.name for p in (REPO / "scripts").glob("*.py")}
    stale = referenced - real - known_scripts - {"app.py"}
    assert not stale, f"docs/architecture.md references non-existent files: {stale}"


def test_doc_cross_links_resolve():
    """Relative links between docs must point at files that exist."""
    import re

    broken = []
    for doc in DOCS.glob("*.md"):
        for target in re.findall(r"\]\((?!https?://|#)([^)]+)\)", doc.read_text()):
            if not (DOCS / target.split("#")[0]).exists():
                broken.append(f"{doc.name} -> {target}")
    assert not broken, f"broken doc links: {broken}"


def test_index_lists_every_doc():
    index = (DOCS / "README.md").read_text()
    missing = [p.name for p in DOCS.glob("*.md")
               if p.name != "README.md" and p.name not in index]
    assert not missing, f"docs/README.md index is missing: {missing}"
