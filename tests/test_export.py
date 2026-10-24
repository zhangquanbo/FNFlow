"""Publication transformations must preserve executable Python semantics."""
from scripts.export_github import ROOT, scrub, portable_python


def test_path_scrubbing_keeps_fstrings_literal():
    source = 'value = f' + repr(str(ROOT) + '/docs/report.md')
    namespace = {}
    exec(compile(scrub(source), '<publication-test>', 'exec'), namespace)
    assert namespace['value'] == '$FNFLOW_ROOT/docs/report.md'

def test_exported_path_calls_are_project_relative():
    from pathlib import Path
    namespace={'__file__':str(Path('/portable/repo/scripts/prepare.py'))}
    exec(compile(portable_python('from pathlib import Path\nROOT=Path("$FNFLOW_ROOT")\nDATA=Path("$FNFLOW_ROOT/data/a")'),'<portable-path>','exec'),namespace)
    assert namespace['DATA']==namespace['ROOT']/'data/a'
    assert '$FNFLOW_ROOT' not in str(namespace['ROOT'])

def test_portable_path_rewrite_preserves_quoted_examples():
    source='example = '+repr('Path("$FNFLOW_ROOT/data/a")')
    assert portable_python(source)==source
