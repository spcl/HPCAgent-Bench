"""Write anonymized copies of SQLite databases for the anonymous mirror; the sources stay untouched.

    python tools/anonymize_dbs.py --out /path/to/anon experiments

Each ``*.db`` under the given roots is copied to ``<out>/<same relative path>``, every text value is
rewritten with the terms of ``.anonymize-terms.txt`` (``pattern=>replacement``; a bare pattern becomes
``XXXX``), and the copy is vacuumed so no old page keeps the original text. The run fails if any term
still matches the raw bytes of a copy.
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
        terms.append((re.compile(pattern, re.IGNORECASE), replacement or DEFAULT_REPLACEMENT))
    return terms


def scrub(text: str, terms: list[tuple[re.Pattern[str], str]]) -> str:
    """Apply every term to one value."""
    for pattern, replacement in terms:
        text = pattern.sub(replacement, text)
    return text


def anonymize(source: pathlib.Path, target: pathlib.Path, terms: list[tuple[re.Pattern[str], str]]) -> None:
    """Copy one database to target and rewrite every text value in the copy."""
    target.parent.mkdir(parents=True, exist_ok=True)
    target.unlink(missing_ok=True)
    with (
        sqlite3.connect(f"file:{source}?mode=ro", uri=True) as src,
        sqlite3.connect(target) as dst,
    ):
        src.backup(dst)
    connection = sqlite3.connect(target)
    connection.create_function("anon", 1, lambda value: scrub(value, terms), deterministic=True)
    tables = [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    for table in tables:
        for column in connection.execute(f'PRAGMA table_info("{table}")').fetchall():
            name = column[1]
            connection.execute(f'UPDATE "{table}" SET "{name}" = anon("{name}") WHERE typeof("{name}") = \'text\'')
    connection.commit()
    connection.execute("VACUUM")
    connection.close()


def leaks(target: pathlib.Path, terms: list[tuple[re.Pattern[str], str]]) -> list[str]:
    """Terms that still match the raw bytes of a copy."""
    raw = target.read_bytes().decode("latin-1")
    return [pattern.pattern for pattern, replacement in terms if pattern.search(raw)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("roots", nargs="+", type=pathlib.Path, help="directories searched for *.db")
    parser.add_argument("--out", required=True, type=pathlib.Path, help="directory for the copies")
    parser.add_argument("--terms", type=pathlib.Path, default=pathlib.Path(".anonymize-terms.txt"))
    args = parser.parse_args()
    terms = load_terms(args.terms)
    out = args.out.resolve()
    failed = False
    for root in args.roots:
        for source in sorted(root.rglob("*.db")):
            source = source.resolve()
            if out in source.parents:
                continue
            target = out / source.relative_to(pathlib.Path.cwd().resolve())
            anonymize(source, target, terms)
            found = leaks(target, terms)
            failed |= bool(found)
            print(f"{'LEAK' if found else 'ok'}\t{target}\t{' '.join(found)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
