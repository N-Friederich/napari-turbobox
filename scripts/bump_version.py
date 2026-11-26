#!/usr/bin/env python3
"""Set the release version across every file that must agree on release day.

Updates five files together and checks every expectation before writing:

    setup.cfg           version = X.Y.Z
    CITATION.cff        version: X.Y.Z and date-released: YYYY-MM-DD
    docs/conf.py        release = 'X.Y.Z'
    docs/changelog.rst  vX.Y.Z (unreleased) -> vX.Y.Z (YYYY-MM-DD)
    README.md           re-pins GitHub tag links from v<old> to vX.Y.Z

Nothing is committed, tagged or pushed. Review the diff before committing.

Usage:
    python scripts/bump_version.py 0.1.1 --dry-run
    python scripts/bump_version.py 0.1.1
    python scripts/bump_version.py 0.1.1 --date 2026-11-25
"""

from __future__ import annotations

import argparse
import datetime as dt
import difflib
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")
# Version 0.2.0 is reserved/unused because paper_results/ records name 0.2.0
# for the pre-release development state measured in the paper.
FORBIDDEN_VERSIONS = {"0.2.0"}

README_PIN_PATTERN = re.compile(
    r"((?:https://raw\.githubusercontent\.com/N-Friederich/napari-turbobox"
    r"|https://github\.com/N-Friederich/napari-turbobox/(?:blob|tree))/v)"
    r"(\d+\.\d+\.\d+)"
    r"(/)"
)


class BumpError(RuntimeError):
    """Raised when a target file does not match the expected release format."""


def _sub_once(text: str, pattern: str, replacement: str, what: str) -> str:
    """Apply exactly one regex substitution, or fail loudly."""
    matches = list(re.finditer(pattern, text, flags=re.M))
    if len(matches) != 1:
        raise BumpError(f"{what}: expected exactly one match for {pattern!r}, found {len(matches)}")
    return re.sub(pattern, replacement, text, count=1, flags=re.M)


def read_current_version(root: Path) -> str:
    """Return the current version string from setup.cfg."""
    path = root / "setup.cfg"
    text = path.read_text(encoding="utf-8")
    matches = re.findall(r"^version[ \t]*=[ \t]*(\d+\.\d+\.\d+)[ \t]*$", text, flags=re.M)
    if len(matches) != 1:
        raise BumpError(f"setup.cfg: expected one 'version = X.Y.Z' line, found {len(matches)}")
    return matches[0]


def bump_setup_cfg(root: Path, old_version: str, version: str) -> tuple[Path, str, str]:
    path = root / "setup.cfg"
    old_text = path.read_text(encoding="utf-8")
    new_text = _sub_once(
        old_text,
        rf"^version[ \t]*=[ \t]*{re.escape(old_version)}[ \t]*$",
        f"version = {version}",
        "setup.cfg version",
    )
    return path, old_text, new_text


def bump_citation(root: Path, version: str, date: str) -> tuple[Path, str, str]:
    path = root / "CITATION.cff"
    old_text = path.read_text(encoding="utf-8")
    new_text = _sub_once(old_text, r"^version:[ \t]*.+$", f"version: {version}", "CITATION.cff version")
    new_text = _sub_once(
        new_text,
        r"^date-released:[ \t]*.+$",
        f"date-released: {date}",
        "CITATION.cff date-released",
    )
    return path, old_text, new_text


def bump_docs_conf(root: Path, version: str) -> tuple[Path, str, str]:
    path = root / "docs" / "conf.py"
    old_text = path.read_text(encoding="utf-8")
    new_text = _sub_once(
        old_text,
        r"^release[ \t]*=[ \t]*['\"][^'\"]+['\"][ \t]*$",
        f"release = '{version}'",
        "docs/conf.py release",
    )
    return path, old_text, new_text


def bump_changelog(root: Path, version: str, date: str) -> tuple[Path, str, str]:
    path = root / "docs" / "changelog.rst"
    old_text = path.read_text(encoding="utf-8")
    if re.search(rf"^v{re.escape(version)} \(\d{{4}}-\d{{2}}-\d{{2}}\)[ \t]*$", old_text, flags=re.M):
        raise BumpError(f"docs/changelog.rst already has a dated section for v{version}")
    header = f"v{version} ({date})"
    underline = "-" * len(header)
    new_text = _sub_once(
        old_text,
        rf"^v{re.escape(version)} \(unreleased\)[ \t]*\n-+[ \t]*$",
        f"{header}\n{underline}",
        f"docs/changelog.rst v{version} (unreleased) heading",
    )
    return path, old_text, new_text


def bump_readme_pins(root: Path, old_version: str, version: str) -> tuple[Path, str, str, int]:
    path = root / "README.md"
    old_text = path.read_text(encoding="utf-8")
    found = [m.group(2) for m in README_PIN_PATTERN.finditer(old_text)]
    if not found:
        raise BumpError("README.md: found no tag-pinned GitHub links to update")
    unexpected = sorted({v for v in found if v != old_version})
    if unexpected:
        raise BumpError(
            f"README.md: expected all pinned links to reference v{old_version}, found {unexpected}"
        )
    new_text, count = README_PIN_PATTERN.subn(rf"\g<1>{version}\g<3>", old_text)
    if count != len(found):
        raise BumpError(f"README.md: expected {len(found)} substitutions, performed {count}")
    return path, old_text, new_text, count


def compute_edits(
    root: Path, version: str, date: str
) -> tuple[list[tuple[Path, str, str]], int]:
    """Validate inputs and compute all five file rewrites without writing."""
    if not VERSION_PATTERN.match(version):
        raise BumpError(f"{version!r} is not a three-part version like 0.1.1")
    if version in FORBIDDEN_VERSIONS:
        raise BumpError(
            f"Version {version} is reserved by benchmark records in paper_results/ and must not be used"
        )
    try:
        dt.date.fromisoformat(date)
    except ValueError as exc:
        raise BumpError(f"{date!r} is not a date in YYYY-MM-DD form") from exc

    old_version = read_current_version(root)
    if version == old_version:
        raise BumpError(f"setup.cfg is already at version {version}")

    setup_edit = bump_setup_cfg(root, old_version, version)
    citation_edit = bump_citation(root, version, date)
    conf_edit = bump_docs_conf(root, version)
    changelog_edit = bump_changelog(root, version, date)
    readme_path, readme_old, readme_new, pin_count = bump_readme_pins(root, old_version, version)
    edits = [
        setup_edit,
        citation_edit,
        conf_edit,
        changelog_edit,
        (readme_path, readme_old, readme_new),
    ]
    return edits, pin_count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("version", help="target release version, e.g. 0.1.1")
    parser.add_argument(
        "--date",
        default=dt.date.today().isoformat(),
        help="release date as YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show planned changes and unified diff without modifying any file",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="repository root (default: parent of scripts/)",
    )
    args = parser.parse_args(argv)
    root = args.root.resolve()

    try:
        edits, pin_count = compute_edits(root, args.version, args.date)
    except BumpError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.dry_run:
        print(f"dry-run: would set version {args.version} (date {args.date}, {pin_count} README links)")
        for path, old_text, new_text in edits:
            rel = path.relative_to(root).as_posix()
            print(f"would write {rel}")
            diff = difflib.unified_diff(
                old_text.splitlines(keepends=True),
                new_text.splitlines(keepends=True),
                fromfile=f"a/{rel}",
                tofile=f"b/{rel}",
                n=1,
            )
            sys.stdout.writelines(diff)
        return 0

    for path, _old_text, new_text in edits:
        path.write_text(new_text, encoding="utf-8", newline="\n")

    print(f"set version {args.version}, date {args.date} ({pin_count} README links updated)\n")
    if (root / ".git").exists():
        subprocess.run(
            ["git", "--no-pager", "diff", "--", *[str(p) for p, _, _ in edits]],
            cwd=root,
            check=False,
        )
    print(
        "\nNothing was committed or tagged. Review the diff, then:\n"
        f"    git commit -am 'Release {args.version}'\n"
        f"    git tag -a v{args.version} -m 'napari-TurboBox {args.version}'\n"
        f"    git push <remote-url> main v{args.version}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
