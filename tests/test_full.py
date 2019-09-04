import os
from os.path import join as joinp
import shutil

from mock import patch
from nose.tools import assert_raises, ok_
from numpy.testing import assert_equal

from .utils import setup_tempdir
from morph_repair.full import full


DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')
def assert_output_exists(root_dir,
                         raw_dir=None,
                         raw_planes_dir=None,
                         unravelled_dir=None,
                         unravelled_planes_dir=None,
                         repaired_dir=None):
    raw_dir = raw_dir or joinp(root_dir, 'raw')
    raw_planes_dir = raw_planes_dir or joinp(raw_dir, 'planes')
    unravelled_dir = unravelled_dir or joinp(root_dir, 'unravelled')
    unravelled_planes_dir = unravelled_planes_dir or joinp(unravelled_dir, 'planes')
    repaired_dir = repaired_dir or joinp(root_dir, 'repaired')

    for folder in [raw_dir, raw_planes_dir, unravelled_dir, unravelled_planes_dir, repaired_dir]:
        ok_(os.path.exists(folder), '{} does not exists'.format(folder))
        ok_(os.listdir(folder), '{} is empty !'.format(folder))

# Patch this to speed up test
@patch('morph_repair.main.plot_repaired_neuron')
@patch('morph_repair.view.view_all')
def test_full(mock1, mock2):
    with setup_tempdir('test-cli-full') as tmp_folder:
        test_folder = joinp(tmp_folder, 'test-full-repair')
        shutil.copytree(joinp(DATA_PATH, 'test-full-repair'), test_folder)
        full(test_folder)
        assert_output_exists(test_folder)


# Patch this to speed up test
@patch('morph_repair.main.plot_repaired_neuron')
@patch('morph_repair.view.view_all')
def test_full_custom_raw_dir(mock1, mock2):
    with setup_tempdir('test-cli-full') as tmp_folder:
        test_folder = joinp(tmp_folder, 'test-full-repair')
        shutil.copytree(joinp(DATA_PATH, 'test-full-repair'), test_folder)

        raw_dir_custom_path = joinp(test_folder, 'raw_custom')
        # Should raise because raw_custom dir does not exist
        assert_raises(Exception, full, test_folder, raw_dir=raw_dir_custom_path)

        shutil.move(joinp(test_folder, 'raw'), raw_dir_custom_path)
        full(test_folder, raw_dir=raw_dir_custom_path)
        assert_output_exists(test_folder, raw_dir=raw_dir_custom_path)


# Patching this to speed up test
@patch('morph_repair.main.plot_repaired_neuron')
@patch('morph_repair.view.view_all')
def test_full_custom_unravel_dir(mock1, mock2):
    with setup_tempdir('test-cli-full') as tmp_folder:
        test_folder = joinp(tmp_folder, 'test-full-repair')
        shutil.copytree(joinp(DATA_PATH, 'test-full-repair'), test_folder)

        custom_path = joinp(test_folder, 'unravel_custom')
        full(test_folder, unravelled_dir=custom_path)
        assert_output_exists(test_folder, unravelled_dir=custom_path)


# Patching this to speed up test
@patch('morph_repair.main.plot_repaired_neuron')
@patch('morph_repair.view.view_all')
def test_full_custom_unravelled_planes_dir(mock1, mock2):
    with setup_tempdir('test-cli-full') as tmp_folder:
        test_folder = joinp(tmp_folder, 'test-full-repair')
        shutil.copytree(joinp(DATA_PATH, 'test-full-repair'), test_folder)

        custom_path = joinp(test_folder, 'unravelled_planes_custom')
        full(test_folder, unravelled_planes_dir=custom_path)
        assert_output_exists(test_folder, unravelled_planes_dir=custom_path)


# Patching this to speed up test
@patch('morph_repair.main.plot_repaired_neuron')
@patch('morph_repair.view.view_all')
def test_full_custom_repaired_dir(mock1, mock2):
    with setup_tempdir('test-cli-full') as tmp_folder:
        test_folder = joinp(tmp_folder, 'test-full-repair')
        shutil.copytree(joinp(DATA_PATH, 'test-full-repair'), test_folder)

        custom_path = joinp(test_folder, 'repaired_planes_custom')
        full(test_folder, repaired_dir=custom_path)
        assert_output_exists(test_folder, repaired_dir=custom_path)


def test_full_custom_plots_dir():
    with setup_tempdir('test-cli-full') as tmp_folder:
        test_folder = joinp(tmp_folder, 'test-full-repair')
        shutil.copytree(joinp(DATA_PATH, 'test-full-repair'), test_folder)

        custom_path = joinp(test_folder, 'plots_custom')
        full(test_folder, plots_dir=custom_path)
        assert_output_exists(test_folder)
        assert_equal(len(os.listdir(custom_path)), 3)
        ok_(os.path.exists(os.path.join(custom_path, 'report.pdf')))
