"""Verify/import already-authorized motion files. Never downloads or accepts terms."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def destination(root, record):
    base = Path(root).resolve()
    target = (base / record['destination']).resolve()
    if not target.is_relative_to(base):
        raise ValueError('Motion destination must remain inside the repository')
    return target

def check(root, records):
    results = []
    for record in records:
        target = destination(root, record)
        state = 'missing' if not target.is_file() else (
            'ok' if digest(target) == record['reference_sha256'] else 'hash_mismatch')
        results.append({'file': record['destination'], 'status': state})
    return results

def import_authorized(source, root, records):
    source = Path(source).resolve()
    if not source.is_dir():
        raise ValueError('Provide a directory containing your independently authorized data')
    planned = []
    for record in records:
        target = destination(root, record)
        matches = [p for p in source.rglob(target.name)
                   if p.is_file() and not p.is_symlink() and digest(p) == record['reference_sha256']]
        if not matches:
            raise ValueError(f'No exact reference match for {target.name}; see README version notes')
        if target.exists() and (not target.is_file() or digest(target) != record['reference_sha256']):
            raise FileExistsError(f'Refusing to overwrite a different file: {target.name}')
        planned.append((matches[0], target))
    for src, target in planned:
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, target)
    return len(planned)

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, help='Your authorized local data directory; nothing is downloaded')
    args = parser.parse_args(argv)
    records = json.loads((ROOT / 'docs/restricted-motions.json').read_text(encoding='utf-8'))['motions']
    if args.source_dir:
        try:
            import_authorized(args.source_dir, ROOT, records)
        except (ValueError, OSError) as exc:
            parser.exit(2, f'{exc}\n')
    results = check(ROOT, records)
    print(json.dumps(results, indent=2))
    if any(item['status'] != 'ok' for item in results):
        print('Restricted reference motions are not ready. Follow README.md; no fallback data is substituted.')
        return 2
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
