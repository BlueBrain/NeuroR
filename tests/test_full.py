import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from mock import patch
from nose.tools import assert_raises, ok_
from numpy.testing import assert_equal

from neuror.full import full

DATA_PATH = Path(__file__).parent / 'data'


def assert_output_exists(root_dir,
                         raw_dir=None,
                         raw_planes_dir=None,
                         unravelled_dir=None,
                         unravelled_planes_dir=None,
                         repaired_dir=None):
    raw_dir = raw_dir or Path(root_dir, 'raw')
    raw_planes_dir = raw_planes_dir or Path(raw_dir, 'planes')
    unravelled_dir = unravelled_dir or Path(root_dir, 'unravelled')
    unravelled_planes_dir = unravelled_planes_dir or Path(unravelled_dir, 'planes')
    repaired_dir = repaired_dir or Path(root_dir, 'repaired')

    for folder in [raw_dir, raw_planes_dir, unravelled_dir, unravelled_planes_dir, repaired_dir]:
        ok_(folder.exists(), '{} does not exists'.format(folder))
        ok_(list(folder.iterdir()), '{} is empty !'.format(folder))


def test_full():
    with TemporaryDirectory('test-full') as tmp_folder:
        test_folder = Path(tmp_folder, 'test-full-repair')
        shutil.copytree(DATA_PATH / 'test-full-repair', test_folder)
        full(test_folder)
        assert_output_exists(test_folder)


def test_full_custom_raw_dir():
    with TemporaryDirectory('test-full-custom-raw-dir') as tmp_folder:
        test_folder = Path(tmp_folder, 'test-full-repair')
        shutil.copytree(DATA_PATH / 'test-full-repair', test_folder)

        raw_dir_custom_path = test_folder / 'raw_custom'
        # Should raise because raw_custom dir does not exist
        assert_raises(Exception, full, test_folder, raw_dir=raw_dir_custom_path)

        shutil.move(test_folder / 'raw', raw_dir_custom_path)
        full(test_folder, raw_dir=raw_dir_custom_path)
        assert_output_exists(test_folder, raw_dir=raw_dir_custom_path)


def test_full_custom_unravel_dir():
    with TemporaryDirectory('test-full-custum-unravel-dir') as tmp_folder:
        test_folder = Path(tmp_folder, 'test-full-repair')
        shutil.copytree(DATA_PATH / 'test-full-repair', test_folder)

        custom_path = test_folder / 'unravel_custom'
        full(test_folder, unravelled_dir=custom_path)
        assert_output_exists(test_folder, unravelled_dir=custom_path)


def test_full_custom_unravelled_planes_dir():
    with TemporaryDirectory('test-full-custom-unravelled-planes-dir') as tmp_folder:
        test_folder = Path(tmp_folder, 'test-full-repair')
        shutil.copytree(DATA_PATH / 'test-full-repair', test_folder)

        custom_path = test_folder / 'unravelled_planes_custom'
        full(test_folder, unravelled_planes_dir=custom_path)
        assert_output_exists(test_folder, unravelled_planes_dir=custom_path)


def test_full_custom_repaired_dir():
    with TemporaryDirectory('test-full-custom-repaired-dir') as tmp_folder:
        test_folder = Path(tmp_folder, 'test-full-repair')
        shutil.copytree(DATA_PATH / 'test-full-repair', test_folder)

        custom_path = test_folder / 'repaired_planes_custom'
        full(test_folder, repaired_dir=custom_path)
        assert_output_exists(test_folder, repaired_dir=custom_path)
