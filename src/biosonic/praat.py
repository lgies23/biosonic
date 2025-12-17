from pathlib import Path
from typing import Union

import textgrids


def _read_textgrid(
        filepath: Union[str, Path]
    ) -> textgrids.TextGrid:
    """
    """
    filepath = Path(filepath)
    grid = textgrids.TextGrid(filepath)

    return grid
