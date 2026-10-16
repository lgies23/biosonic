import biosonic


def test_version_and_version_tuple():
    assert hasattr(biosonic, '__version__')
    assert hasattr(biosonic, '__version_tuple__')
    assert biosonic.__version__ != "0.0.0"
    assert biosonic.__version_tuple__ != (0, 0, 0)
    assert biosonic.__version_tuple__[:3] == tuple(map(int, biosonic.__version__.split('.')[:3]))
