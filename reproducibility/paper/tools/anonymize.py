"""Write anonymized copies of SQLite databases and text files for the anonymous release; the sources stay untouched.

    python tools/anonymize.py --out /path/to/anon <dir>...

Every file under the given roots is copied: a ``*.db`` as a database, any other file as UTF-8 text,
and a binary unchanged to ``<out>/<same relative path>``, every text value is
rewritten with the terms of ``.anonymize-terms.txt`` (``pattern=>replacement``; a bare pattern becomes
``XXXX``), and a database copy is vacuumed so no old page keeps the original text. The run fails if any term
still matches the raw bytes of a copy, so a binary that carries an identifier stops the release.
"""

import argparse
import pathlib
import re
import sqlite3
import sys

DEFAULT_REPLACEMENT = "XXXX"
#: Terms made of letters only match whole words, so "ETH" does not hit "method".
WORD_TERM = re.compile(r"[A-Za-z\[\] ]+")
#: Account names such as "g34" must not match inside other tokens or record bytes ("Bg341.3").
ACCOUNT_TERM = re.compile(r"[A-Za-z-]+[0-9]+")


def load_terms(path: pathlib.Path) -> list[tuple[re.Pattern[str], str]]:
    """Compiled (pattern, replacement) pairs, case-insensitive, in file order."""
    terms = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        pattern, arrow, replacement = line.partition("=>")
        if WORD_TERM.fullmatch(pattern):
            pattern = rf"\b{pattern}\b"
        elif ACCOUNT_TERM.fullmatch(pattern):
            pattern = rf"(?<![A-Za-z0-9]){pattern}(?![0-9])"
        terms.append(
            (re.compile(pattern, re.IGNORECASE), replacement or DEFAULT_REPLACEMENT)
        )
    return terms


def scrub(text: str, terms: list[tuple[re.Pattern[str], str]]) -> str:
    """Apply every term to one value."""
    for pattern, replacement in terms:
        text = pattern.sub(replacement, text)
    return text


def anonymize(
    source: pathlib.Path, target: pathlib.Path, terms: list[tuple[re.Pattern[str], str]]
) -> None:
    """Copy one database to target and rewrite every text value in the copy."""
    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)
    with (
        sqlite3.connect(f"file:{source}?mode=ro", uri=True) as src,
        sqlite3.connect(target) as dst,
    ):
        src.backup(dst)
    connection = sqlite3.connect(target)
    connection.create_function(
        "anon", 1, lambda value: scrub(value, terms), deterministic=True
    )
    tables = [
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    ]
    for table in tables:
        for column in connection.execute(f'PRAGMA table_info("{table}")').fetchall():
            name = column[1]
            connection.execute(
                f'UPDATE "{table}" SET "{name}" = anon("{name}") WHERE typeof("{name}") = \'text\''
            )
    connection.commit()
    connection.execute("VACUUM")
    connection.close()


def leaks(target: pathlib.Path, terms: list[tuple[re.Pattern[str], str]]) -> list[str]:
    """Terms that still match the raw bytes of a copy."""
    raw = target.read_bytes().decode("latin-1")
    return [pattern.pattern for pattern, replacement in terms if pattern.search(raw)]


def check(roots: list[pathlib.Path], terms: list[tuple[re.Pattern[str], str]]) -> int:
    """Report every file under ``roots`` whose raw bytes a term still matches; 1 if any does."""
    found = 0
    for root in roots:
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            hits = leaks(path, terms)
            if hits:
                found += 1
                print(f"LEAK\t{path}\t{' '.join(hits)}")
    print(
        f"{'clean' if not found else f'{found} file(s) leak'}: {sum(1 for r in roots for p in r.rglob('*') if p.is_file())} files checked"
    )
    return 1 if found else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "roots",
        nargs="+",
        type=pathlib.Path,
        help="directories whose files are anonymized",
    )
    parser.add_argument("--out", type=pathlib.Path, help="directory for the copies")
    parser.add_argument(
        "--check",
        action="store_true",
        help="write nothing; report every file a term still matches",
    )
    parser.add_argument(
        "--terms", type=pathlib.Path, default=pathlib.Path(".anonymize-terms.txt")
    )
    args = parser.parse_args()
    terms = load_terms(args.terms)
    if args.check:
        return check(args.roots, terms)
    if args.out is None:
        parser.error("--out is required unless --check")
    out = args.out.resolve()
    failed = False
    for root in args.roots:
        for source in sorted(root.rglob("*")):
            source = source.resolve()
            if out in source.parents or not source.is_file():
                continue
            target = out / source.relative_to(pathlib.Path.cwd().resolve())
            if source.suffix == ".db":
                anonymize(source, target, terms)
            elif source.suffix in (".db-shm", ".db-wal"):
                continue
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    target.write_text(
                        scrub(source.read_text(encoding="utf-8"), terms),
                        encoding="utf-8",
                    )
                except UnicodeDecodeError:
                    target.write_bytes(
                        source.read_bytes()
                    )  # a binary (figure) is copied, then checked below
            found = leaks(target, terms)
            failed |= bool(found)
            print(f"{'LEAK' if found else 'ok'}\t{target}\t{' '.join(found)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
