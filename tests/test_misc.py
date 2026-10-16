import importlib.metadata
import json

import biosonic


def test_version_and_version_tuple():
    direct_url = importlib.metadata.Distribution.from_name("biosonic").read_text("direct_url.json")
    is_editably_installed = json.loads(direct_url).get("dir_info", {}).get("editable", False)

    assert hasattr(biosonic, '__version__')
    assert hasattr(biosonic, '__version_tuple__')
    if is_editably_installed:
        assert biosonic.__version__ == "0.0.0"
        assert biosonic.__version_tuple__ == (0, 0, 0)
    else:
        assert biosonic.__version__ != "0.0.0"
        assert biosonic.__version_tuple__ != (0, 0, 0)
        assert biosonic.__version_tuple__[:3] == tuple(map(int, biosonic.__version__.split('.')[:3]))
