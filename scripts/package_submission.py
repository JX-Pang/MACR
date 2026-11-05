from __future__ import annotations
import argparse
import re
import zipfile
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / 'dist' / 'macr_aaai_anonymous.zip'
INCLUDED_ROOTS = ('macr', 'scripts', 'configs')
INCLUDED_FILES = ('.gitignore', 'README.md', 'requirements.txt')
TEXT_SUFFIXES = {'.py', '.yaml', '.yml', '.md', '.txt', ''}
SENSITIVE_PATTERNS = {'Windows user directory': re.compile('[A-Za-z]:[\\\\/](?:Users|Documents and Settings)[\\\\/][^\\\\/\\s]+', re.IGNORECASE), 'Unix home directory': re.compile('/h[o]me/[^/\\s]+/'), 'macOS user directory': re.compile('/U[s]ers/[^/\\s]+/'), 'private key block': re.compile('BEGIN [A-Z ]*PRIVATE KEY'), 'common API token': re.compile('\\b(?:sk|xox[baprs])-[A-Za-z0-9_-]{12,}\\b'), 'assigned credential': re.compile('(?i)\\b(?:api[_-]?key|access[_-]?token|password|secret)\\b\\s*[:=]\\s*[\\"\'][^\\"\']{8,}[\\"\']')}

def collect_files() -> list[Path]:
    files: set[Path] = set()
    for root_name in INCLUDED_ROOTS:
        root = PROJECT_ROOT / root_name
        files.update((path for path in root.rglob('*') if path.is_file() and '__pycache__' not in path.parts))
    for name in INCLUDED_FILES:
        path = PROJECT_ROOT / name
        if not path.is_file():
            raise FileNotFoundError(f'Required release file is missing: {name}')
        files.add(path)
    return sorted(files, key=lambda path: path.relative_to(PROJECT_ROOT).as_posix())

def scan_text(path: Path) -> list[str]:
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return []
    text = path.read_text(encoding='utf-8')
    return [label for label, pattern in SENSITIVE_PATTERNS.items() if pattern.search(text)]

def write_deterministic_zip(files: list[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            relative = path.relative_to(PROJECT_ROOT).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 420 << 16
            archive.writestr(info, path.read_bytes())

def main() -> None:
    parser = argparse.ArgumentParser(description='Build an anonymous, code-only MACR submission archive.')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    files = collect_files()
    findings = {path.relative_to(PROJECT_ROOT).as_posix(): scan_text(path) for path in files}
    findings = {path: labels for path, labels in findings.items() if labels}
    if findings:
        details = '; '.join((f"{path}: {', '.join(labels)}" for path, labels in sorted(findings.items())))
        raise RuntimeError(f'Sensitive content found; archive not created: {details}')
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    write_deterministic_zip(files, output)
    print(f'Created {output.relative_to(PROJECT_ROOT).as_posix()} with {len(files)} files')
if __name__ == '__main__':
    main()
