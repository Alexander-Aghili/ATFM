"""Build the standalone background paper without touching the research manuscript."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NAME = 'llm-serving-background'


def run(args, **kwargs):
    subprocess.run(args, check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figures', action='store_true')
    parser.add_argument('--bundle', action='store_true')
    args = parser.parse_args()
    run([sys.executable, str(HERE/'make_sources.py')])
    if args.figures:
        _figures()
    output = _compile()
    if args.bundle:
        _bundle()
    print(output/f'{NAME}.pdf')


def _bundle():
    paths = [p for p in HERE.rglob('*') if p.is_file() and
             p.suffix in {'.tex', '.pdf', '.png', '.svg', '.reladraw', '.py', '.md', '.json'}
             and p.name != f'{NAME}.pdf']
    with zipfile.ZipFile(HERE/f'{NAME}-source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            archive.write(path, path.relative_to(HERE))


def _compile():
    scratch = ROOT/'tmp/pdfs'/NAME
    scratch.mkdir(parents=True, exist_ok=True)
    with (scratch/'build.log').open('w') as log:
        run(['latexmk', '-pdf', '-interaction=nonstopmode', '-halt-on-error',
             f'-outdir={scratch}', f'{NAME}.tex'], cwd=HERE,
            stdout=log, stderr=subprocess.STDOUT)
    _validate_log(scratch)
    output = ROOT/'output/pdf'
    output.mkdir(parents=True, exist_ok=True)
    for directory in (HERE, output):
        shutil.copyfile(scratch/f'{NAME}.pdf', directory/f'{NAME}.pdf')
    return output


def _validate_log(scratch):
    log = (scratch/f'{NAME}.log').read_text(errors='replace')
    issues = [line for line in log.splitlines() if any(term in line for term in
              ('Overfull', 'undefined', 'Missing character', 'multiply defined'))]
    if issues:
        raise SystemExit('\n'.join(issues))


def _figures():
    run([sys.executable, str(HERE/'plots.py')])
    run([sys.executable, str(HERE/'evidence_plots.py')])
    for source in sorted((HERE/'figures').glob('*.reladraw')):
        svg = source.with_suffix('.svg')
        run(['npx', '--yes', 'reladraw', str(source), '-o', str(svg)])
        run(['inkscape', str(svg), '--export-type=pdf',
             f'--export-filename={source.with_suffix(".pdf")}'],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


if __name__ == '__main__':
    main()
