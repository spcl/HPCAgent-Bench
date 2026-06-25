"""Every optarena-owned YAML follows the one house style.

Guards the repo-wide convention enforced by ``tests/check_yaml_style.py``
(``#`` header line, 2-space structural indent, no tabs, no trailing whitespace,
single final newline). Third-party-schema YAML (GitHub Actions, docker-compose,
Continue.dev) is excluded by the checker itself.
"""
import pathlib

from tests.check_yaml_style import owned_yaml, violations

REPO = pathlib.Path(__file__).resolve().parent.parent


def test_all_owned_yaml_conforms():
    bad = {f.relative_to(REPO): probs for f in owned_yaml() if (probs := violations(f))}
    assert not bad, "YAML style violations (run `python tests/check_yaml_style.py --fix`):\n" + \
        "\n".join(f"{f}: {p}" for f, p in bad.items())
