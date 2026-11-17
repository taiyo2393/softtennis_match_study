"""Make correctly proportioned context sheets; motion clips remain primary evidence."""
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r'C:\Users\ta239\Research\softtennis_match_study\study')
OUT = Path(__file__).resolve().parent
cap = cv2.VideoCapture(str(ROOT / 'data/raw/match1.mp4'))
assert cap.isOpened()
for name, frames in {
    '044_point_end_aspect_correct': [115760, 115772, 115780, 115788, 115796, 115804],
    '056_point_end_aspect_correct': [154830, 154850, 154870, 154890, 154910, 154930],
}.items():
    tiles = []
    for frame_id in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, image = cap.read()
        assert ok, frame_id
        height, width = image.shape[:2]
        assert (width, height) == (3840, 2160)
        tile = cv2.resize(image, (960, 540), interpolation=cv2.INTER_AREA)
        cv2.rectangle(tile, (0, 0), (185, 36), (0, 0, 0), -1)
        cv2.putText(tile, f'F{frame_id}', (10, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
        tiles.append(tile)
    sheet = np.vstack([np.hstack(tiles[i:i+3]) for i in (0, 3)])
    assert cv2.imwrite(str(OUT / f'{name}.jpg'), sheet)
cap.release()
