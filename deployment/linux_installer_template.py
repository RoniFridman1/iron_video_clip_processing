#! /usr/bin/python3
"""
This file is a template for an installer that runs on Linux.

To obtain a full installer, do the following:

1. Updating compressed data to be a base64-encoded .tar.gz file containing
   all files that should be extracted to the file system
2. Updating the setup path to be a path to a file contained in the above
   archive (it will be executed from the root directory)

The setup script will run once extraction completes, receiving all the
arguments passed to the installer.
"""

import base64
import io
import gzip
import sys
import os
import tarfile
import shlex


COMPRESSED_DATA = b"@@TAR_GZ_BASE64@@"
SETUP_PATH = '@@SETUP_PATH@@'


def decompress():
    print('INSTALL: Decompressing')
    tar_gz_bytes = base64.b64decode(COMPRESSED_DATA)
    tar_bytes = gzip.decompress(tar_gz_bytes)

    print('INSTALL: Extracting')
    tar_stream = io.BytesIO(tar_bytes)
    tar_stream.seek(0)
    with tarfile.TarFile(fileobj=tar_stream) as tar:
        tar.extractall()


def maybe_install():
    if not os.path.isfile(SETUP_PATH):
        print(f'INSTALL: No {SETUP_PATH} file found, stopping')
    elif '--extract-only' in sys.argv:
        print('INSTALL: Skipping post-extraction setup since '
            '`--extract-only` was specified')
    else:
        print('INSTALL: Configuring setup script')
        os.chmod(SETUP_PATH, 0o755)
        cmd = [SETUP_PATH] + sys.argv[1:]
        print(f'INSTALL: Running `{shlex.join(cmd)}`')
        os.execv(SETUP_PATH, cmd)


if __name__ == '__main__':
    decompress()
    maybe_install()