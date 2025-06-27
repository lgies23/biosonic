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

    if 'syllables' not in grid:
        print(f'No "syllables" grid in: {filepath}')
        return None
    
    for syll in grid['syllables']:
        # Convert Praat to Unicode in the label
        label = syll.text.transcode()
        # Print label and syllable duration
        print(f'"{label}"; {syll.dur}')
    
    return grid