# Copyright 2021 ETH Zurich and the OptArena authors.
# SPDX-License-Identifier: GPL-3.0-or-later
import difflib
import os
import time

# pygount transitively imports GitPython, which aborts at import time if the
# ``git`` executable is absent (e.g. a minimal container / pip env). OptArena only
# uses pygount for local source line-counting, never its git features, so degrade
# GitPython to a no-op instead of requiring the binary. Must precede the import.
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

from optarena.infrastructure import Benchmark, Framework, utilities as util  # noqa: E402
from pygount import SourceAnalysis  # noqa: E402


class LineCount(object):
    """ A class for counting lines of code. """

    def __init__(self, bench: Benchmark, frmwrk: Framework, npfrmwrk: Framework = None):
        self.bench = bench
        self.frmwrk = frmwrk
        self.numpy = npfrmwrk

    def count(self):
        """ Counts the code lines of the framework's benchmark implementations
        and how many lines are different compared to the NumPy implementation.
        """

        if self.numpy:
            np_file, _ = self.numpy.impl_files(self.bench)[0]
            np_analysis = SourceAnalysis.from_file(np_file, "pygount")
        else:
            np_analysis = None

        # create a database connection
        database = r"optarena.db"
        conn = util.create_connection(database)

        # create tables
        if conn is not None:
            # create results table
            util.create_table(conn, util.sql_create_lcounts_table)
        else:
            print("Error! cannot create the database connection.")

        # Extra information
        kind = ""
        if "kind" in self.bench.info.keys():
            kind = self.bench.info["kind"]
        domain = ""
        if "domain" in self.bench.info.keys():
            domain = self.bench.info["domain"]
        dwarf = ""
        if "dwarf" in self.bench.info.keys():
            dwarf = self.bench.info["dwarf"]
        version = self.frmwrk.version()

        avalues = []
        for impl_file, impl_name in self.frmwrk.impl_files(self.bench):
            try:
                frmwrk_analysis = SourceAnalysis.from_file(impl_file, "pygount")
                if np_analysis:
                    text1 = open(np_file).readlines()
                    text2 = open(impl_file).readlines()
                    diff = difflib.Differ()

                    changed_lines = 0
                    for line in diff.compare(text1, text2):
                        if line[0] == '-':
                            changed_lines += 1
                    avalues.append({'details': impl_name, 'count': frmwrk_analysis.code_count, 'npdiff': changed_lines})
            except Exception:
                continue

        # Write data
        timestamp = int(time.time())
        for d in avalues:
            new_d = {
                'timestamp': timestamp,
                'benchmark': self.bench.info["short_name"],
                'kind': kind,
                'domain': domain,
                'dwarf': dwarf,
                'mode': "main",
                'framework': self.frmwrk.info["simple_name"],
                'version': version,
                'details': d["details"],
                'count': d["count"],
                'npdiff': d["npdiff"]
            }
            result = tuple(new_d.values())
            util.create_result(conn, util.sql_insert_into_lcounts_table, result)
