import textgrids
from typing import Union
from pathlib import Path


def _read_textgrid(
        filepath: Union[str, Path]
    ) -> textgrids.TextGrid:
    """
    """
    filepath = Path(filepath)
    grid = textgrids.TextGrid(filepath)

    return grid
