import textgrids
from typing import Union, Optional
from pathlib import Path


def read_textgrid(
        filepath : Union[str, Path]
    ) -> Optional[textgrids.TextGrid]:
    """
    """
    filepath = Path(filepath)
    grid = textgrids.TextGrid(filepath)
    
    return grid