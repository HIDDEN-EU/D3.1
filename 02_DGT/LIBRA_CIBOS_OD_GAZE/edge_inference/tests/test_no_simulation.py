"""The no-mocks rule, enforced instead of documented.

The package this replaced was built contract-first around dummy adapters. Those
are gone, and nothing may reintroduce them: a stand-in that emits
contract-shaped data is indistinguishable downstream from a real measurement,
which is exactly the failure mode a safety signal cannot have.

Test stand-ins are fine — they live in tests/ and are never importable from the
package.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "dgt"
# The harness pipelines left the package for scripts/ on 2026-09-25. They still
# drive real models on the device, so the same rule follows them there.
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"

# Names that would indicate a stand-in implementation inside the package.
BANNED = (
    "SimulationGazeModel",
    "simulation_model",
    "build_model",
    "DummyGaze",
    "DummyEmotion",
    "MagicMock",
)

_COMMENT = re.compile(r"\s*#")


def _code_lines(path: Path):
    """Yield (lineno, line) for lines that are neither comments nor docstring text.

    Crude but sufficient: the package's docstrings legitimately DISCUSS the
    simulation backend to explain why it is unreachable, so prose must not trip
    the check.
    """
    in_doc = False
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.strip()
        triples = stripped.count('"""') + stripped.count("'''")
        if in_doc:
            if triples:
                in_doc = False
            continue
        if triples == 1:
            in_doc = True
            continue
        if triples >= 2:
            continue
        if not stripped or _COMMENT.match(line):
            continue
        yield number, line


@pytest.mark.parametrize(
    "path",
    sorted(PACKAGE.rglob("*.py")) + sorted(SCRIPTS.glob("*.py")),
    ids=lambda p: f"{p.parent.name}/{p.name}",
)
def test_no_stand_in_implementation_in_the_package(path: Path):
    offenders = [
        f"{path.relative_to(PACKAGE.parent)}:{number}: {line.strip()}"
        for number, line in _code_lines(path)
        for name in BANNED
        if name in line
    ]
    assert not offenders, "stand-in implementation referenced in package code:\n" + "\n".join(
        offenders
    )


def test_the_package_ships_no_dummy_or_simulation_module():
    names = [p.name for p in PACKAGE.rglob("*.py")]
    assert not [n for n in names if "dummy" in n or "simulation" in n or "mock" in n]


def test_the_test_stand_ins_are_not_importable_from_the_package():
    import dgt

    assert not hasattr(dgt, "fakes")
    assert not (PACKAGE / "fakes.py").exists()
