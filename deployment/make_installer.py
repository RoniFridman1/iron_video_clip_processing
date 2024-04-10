import argparse
import base64
import gzip
import io
import pathlib
import tarfile
import sys
from typing import Iterable


IGNORED_SUFFIXES = {
    '.mp4',
    '.pyc',
    '.jpg',
    '.png',
}

IGNORED_FILES = {
    '.gitignore',
}

IGNORED_PATTERNS = {
    '.git',
}

# Paths are relative to the execution dir of this script.
INCLUDE_SOURCES = [
    'ironvideo',
    'requirements.txt',
    'CodeFormer',
    'yolo_tracking',
]


def list_eligible_files(root: pathlib.Path):
    if root.is_file():
        paths = [root]
    else:
        paths = root.rglob('*')
    for f in paths:
        if not f.is_file():
            continue
        if f.suffix in IGNORED_SUFFIXES:
            continue
        if f.name in IGNORED_FILES:
            continue
        if any(p in str(f) for p in IGNORED_PATTERNS):
            continue
        yield f


def create_tar_archive(sources: Iterable[pathlib.Path],
                       base_dir: pathlib.Path) -> bytes:
    tar_stream = io.BytesIO()
    base_dir = base_dir.resolve()
    with tarfile.TarFile(fileobj=tar_stream, mode='w') as tar:
        for source in sources:
            if not source.exists():
                raise FileNotFoundError(f'Could not find {source}!')
            for path in list_eligible_files(source):
                tar.add(path.resolve().relative_to(base_dir))
    return tar_stream.getvalue()


def create_installer_script(*,
                            setup_path: pathlib.Path,
                            template_path: pathlib.Path) -> str:
    base_dir = pathlib.Path('.').resolve()
    tar_archive = create_tar_archive(
        [pathlib.Path(source) for source in INCLUDE_SOURCES] + [setup_path],
        base_dir=base_dir
    )
    rel_setup_path = setup_path.resolve().relative_to(base_dir)
    tar_gz_base64 = base64.b64encode(gzip.compress(tar_archive)).decode()
    return (
        template_path.read_text()
            .replace('@@TAR_GZ_BASE64@@', tar_gz_base64)
            .replace('@@SETUP_PATH@@', str(rel_setup_path))
    )


def create_aws_amazon_linux_installer() -> str:
    deploy_dir = pathlib.Path(__file__).parent
    return create_installer_script(
        setup_path=deploy_dir / 'aws_setup.sh',
        template_path=deploy_dir / 'linux_installer_template.py',
    )


def create_test_installer() -> str:
    deploy_dir = pathlib.Path(__file__).parent
    return create_installer_script(
        setup_path=deploy_dir / 'test_setup.py',
        template_path=deploy_dir / 'linux_installer_template.py',
    )


def main(argv=None):
    parser = argparse.ArgumentParser('Generate an installer script')
    parser.add_argument(
        '--type', choices=['aws-linux', 'test-linux'], default='aws-linux',
        help='The installer type to generate (default: aws-linux)')
    result = parser.parse_args(argv)
    match result.type:
        case 'aws-linux':
            action = create_aws_amazon_linux_installer
            out = 'aws_installer.py'
        case 'test-linux':
            action = create_test_installer
            out = 'test_installer.py'
        case _:
            parser.error(f'ERROR: Unsupported type {result.type}')
    print(f'Creating {result.type} installer')
    with open(out, 'w') as f:
        f.write(action())
    print(f'Saved in {out}')


if __name__ == '__main__':
    main()