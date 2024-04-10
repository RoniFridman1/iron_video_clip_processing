#! /usr/bin/python3
# This file is a test setup script to be run after extracting the files from
# the installer. It will log the current directory and arguments and then exit.
import sys
import pathlib
import os


with (pathlib.Path(__file__).parent / 'test_setup_results.txt').open('w') as f:
    f.write(f'Args: {sys.argv}\n')
    f.write(f'Working dir: {os.getcwd()}\n')
