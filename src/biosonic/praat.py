from pathlib import Path
from typing import Union

try:
    import textgrids
except ModuleNotFoundError:
    raise ImportError("praat-textgrids is required for TextGrid support. Install it with: pip install biosonic[praat]")


def _read_textgrid(
        filepath: Union[str, Path]
    ) -> textgrids.TextGrid:
    """
    """
    filepath = Path(filepath)
    grid = textgrids.TextGrid(filepath)

    return grid
