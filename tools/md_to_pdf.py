"""Render a Markdown document of this repository (GitHub flavour) to PDF with pandoc and XeLaTeX.

python tools/md_to_pdf.py ARCHITECTURE.md        # writes ARCHITECTURE.pdf next to it

GitHub-only syntax is rewritten first: ```math blocks become display math, the fire and snowflake emoji become the
same Font Awesome icons the figures use, and centred <img> tags become pandoc images.
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HEADER=r'''\usepackage{fontawesome5}
\usepackage{xcolor}
\definecolor{fzline}{HTML}{3B7FC4}\definecolor{trline}{HTML}{D9730D}
\newcommand{\fire}{{\color{trline}\faFire}}
\newcommand{\snow}{{\color{fzline}\faSnowflake}}
\usepackage{amsmath,amssymb}
\usepackage{float}\floatplacement{figure}{H}
'''


def rewrite(text):
    text=re.sub(r'```math\n(.*?)\n```',lambda m:'$$\n'+m.group(1).strip()+'\n$$',text,flags=re.S)
    text=text.replace('🔥','\\fire{}').replace('❄️','\\snow{}').replace('❄','\\snow{}')
    text='\n'.join(prose(text.split('\n')))
    text=re.sub(r'\]\(#\d+-',r'](#',text)  # GitHub anchors keep the section number; pandoc's drop it
    text=re.sub(r'<p align="center"><img src="([^"]+)" alt="([^"]*)"[^>]*></p>',r'![\2](\1){width=100%}',text)
    return text


LIST=re.compile(r'(- |\d+\. )')


def prose(lines):
    """Outside code: the arrow the text font lacks becomes TeX, and a list gets the blank line pandoc needs before it."""
    fenced=False;previous=''
    for line in lines:
        if line.startswith('```'):fenced=not fenced
        code=fenced or line.startswith('    ') or line.startswith('```')
        if not code:
            if LIST.match(line) and previous.strip() and not LIST.match(previous) and not previous.startswith(' '):yield ''
            line=line.replace('→','$\\rightarrow$')
        yield line;previous=line


def main(path):
    source=Path(path).resolve();out=source.with_suffix('.pdf')
    with tempfile.TemporaryDirectory() as tmp:
        md=Path(tmp)/source.name;md.write_text(rewrite(source.read_text()))
        header=Path(tmp)/'header.tex';header.write_text(HEADER)
        subprocess.run(['pandoc',str(md),'-o',str(out),'--pdf-engine=xelatex','--resource-path',str(source.parent),
                        '-H',str(header),'-V','geometry:margin=2.2cm','-V','mainfont=Palatino','-V','monofont=Menlo',
                        '-V','fontsize=10pt','-V','colorlinks=true','-V','linkcolor=blue!50!black','-V','urlcolor=blue!50!black',
                        '--toc-depth=2'],check=True,cwd=source.parent)
    print('wrote',out)


if __name__=='__main__':
    main(sys.argv[1])
