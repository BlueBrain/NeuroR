'''Module providing utility functions for the tests'''
import re
import shutil
import sys
import tempfile
from contextlib import contextmanager
from functools import partial
from io import StringIO

from nose.tools import assert_raises, ok_

from morphio import Morphology, set_ignored_warning


@contextmanager
def setup_tempdir(prefix, no_cleanup=False):
    '''Context manager returning a temporary directory'''
    temp_dir = tempfile.mkdtemp(prefix=prefix)
    try:
        yield temp_dir
    finally:
        if not no_cleanup:
            shutil.rmtree(temp_dir)
