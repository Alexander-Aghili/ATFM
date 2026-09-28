"""Compile the editable LaTeX paper; optionally refresh vector figures from frozen results.

python3 docs/paper/build_latex.py [--figures]
A direct latexmk build also works with the checked-in figures and bibliography.
"""
from pathlib import Path
import argparse
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
import zipfile

HERE = Path(__file__).resolve().parent


def figures():
    _refresh_figures()
    dest = HERE / 'latex-fig'
    dest.mkdir(exist_ok=True)
    _chart_svgs(dest)
    shutil.copyfile(HERE/'fig/fig-system.svg', dest/'system.svg')
    for name in ['forecast-flow', 'research-roadmap']:
        subprocess.run(['npx','--yes','reladraw',str(HERE/'fig'/f'{name}.reladraw'),
                        '-o',str(dest/f'{name}.svg')],check=True,stdout=subprocess.DEVNULL)
    shutil.copyfile(HERE/'sources/prior-work/thunderagent-overview.png',dest/'prior-thunderagent.png')
    subprocess.run(['inkscape',str(HERE/'sources/prior-work/continuum-cache.svg'),
                    '--export-type=pdf',f'--export-filename={dest/"prior-continuum.pdf"}'],
                    check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    for path in sorted(dest.glob('*.svg')):
        subprocess.run(['inkscape',str(path),'--export-type=pdf',
                        f'--export-filename={path.with_suffix(".pdf")}'],check=True,
                       stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)


def _chart_svgs(dest):
    import build_paper as paper
    css = _chart_css()
    svgs = {'h1-tracelab':paper.chart_ratio(paper.TRACELAB,'(a) TraceLab'),
            'h1-agentx':paper.chart_ratio(paper.AGENTX,'(b) AgentX'),
            'h1b-progress':paper.chart_h1b(), 'h2-tradeoff':paper.chart_h2()}
    for name, svg in svgs.items():
        svg = svg.replace('<svg ', '<svg xmlns="http://www.w3.org/2000/svg" ', 1)
        svg = svg.replace('>', f'><style>{css}</style>', 1)
        # Inkscape expects explicit page dimensions for predictable PDF export.
        dims = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg)
        svg = svg.replace('<svg ', f'<svg width="{dims[1]}" height="{dims[2]}" ', 1)
        path = dest / f'{name}.svg'
        path.write_text(svg)
        ET.parse(path)


def _chart_css():
    # Standalone SVG styling; the HTML charts normally inherit these rules from paper.css.
    css = '''svg{font-family:Arial,sans-serif} .grid{stroke:#d0d0d0;stroke-width:.7}
.base{stroke:#444;stroke-width:1;stroke-dasharray:4 3}.tick{fill:#444;font-size:14px}
.axis{fill:#333;font-size:14px}.lab{fill:#222;font-size:11.5px}.strong{font-weight:700}
.ct{fill:#171717;font-size:14px;font-weight:700}.val{fill:#222;font-size:10px}
.line{fill:none;stroke-width:2;stroke-linejoin:round}.dot{stroke:white;stroke-width:1}
.bar{stroke:none}.ci{stroke:#333;stroke-width:1.5}
.s1.line{stroke:#245b85}.s2.line{stroke:#b14b20;stroke-dasharray:7 3}
.s3.line{stroke:#35734d;stroke-dasharray:2 3}
.s1.dot,.s1.bar{fill:#245b85}.s2.dot,.s2.bar{fill:#b14b20}.s3.dot,.s3.bar{fill:#35734d}'''
    return css


def _refresh_figures():
    import build_paper as paper
    import build_visuals
    build_visuals.main()
    import probability_visuals
    probability_visuals.main()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figures',action='store_true',help='Regenerate vector figures from saved result summaries')
    parser.add_argument('--bundle',action='store_true',help='Create a portable LaTeX source ZIP')
    args=parser.parse_args()
    if args.figures:
        figures()
    _compile()
    if args.bundle:
        _bundle()


def _bundle():
    paths = [HERE/name for name in ['atfm-paper.tex','references.bib','README.md',
             'build_latex.py','build_visuals.py','probability_visuals.py','VISUALS.md','build_paper.py','paper.css','manuscript.html',
             'fig/forecast-flow.reladraw','fig/research-roadmap.reladraw',
             'fig/fig-system.svg','fig/fig-system.reladraw']]
    paths.append(HERE / 'control-implementation.tex')
    paths += sorted((HERE/'latex-fig').glob('*'))
    paths += sorted((HERE/'results').glob('*'))
    paths += sorted((HERE/'sources/prior-work').glob('*'))
    with zipfile.ZipFile(HERE/'atfm-latex.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            if path.is_file():
                archive.write(path,path.relative_to(HERE))
    print(HERE/'atfm-latex.zip')


def _compile():
    out=HERE/'tmp/latex'
    out.mkdir(parents=True,exist_ok=True)
    run=subprocess.run(['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error',
                        '-file-line-error','-outdir=tmp/latex','atfm-paper.tex'],cwd=HERE,
                       text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if run.returncode:
        print(run.stdout)
        raise SystemExit(run.returncode)
    _validate_log(out)
    shutil.copyfile(out/'atfm-paper.pdf',HERE/'atfm-paper.pdf')
    print(HERE/'atfm-paper.pdf')


def _validate_log(out):
    log=(out/'atfm-paper.log').read_text(errors='replace')
    issues=[line for line in log.splitlines() if any(s in line for s in
            ['Overfull','undefined','Missing character','multiply defined'])]
    if issues:
        print('\n'.join(issues))
        raise SystemExit('Resolve LaTeX layout/reference warnings before publishing the PDF.')


if __name__=='__main__':
    main()
