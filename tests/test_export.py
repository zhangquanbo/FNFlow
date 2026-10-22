"""Publication transformations must preserve executable Python semantics."""
from scripts.export_github import ROOT, scrub


def test_path_scrubbing_keeps_fstrings_literal():
    source = 'value = f' + repr(str(ROOT) + '/docs/report.md')
    namespace = {}
    exec(compile(scrub(source), '<publication-test>', 'exec'), namespace)
    assert namespace['value'] == '$FNFLOW_ROOT/docs/report.md'
