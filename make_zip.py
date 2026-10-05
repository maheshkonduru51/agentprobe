from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / 'agentprobe.zip'

with ZipFile(OUT, 'w', ZIP_DEFLATED) as zf:
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or path.name == 'agentprobe.db':
            continue
        if '__pycache__' in path.parts or path.suffix in {'.pyc'}:
            continue
        zf.write(path, Path('agentprobe') / path.relative_to(ROOT))
print(OUT)
