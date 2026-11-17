"""Extract selected full-rally tracker preview frames for identity review."""
from pathlib import Path
import cv2

OUT = Path(__file__).resolve().parent
samples = {
    '044': (114710,114850,114865,114870,115180,115216,115230,115540,115780),
    '056': (154086,154125,154134,154136,154290,154375,154645,154870),
}
for rally, frames in samples.items():
    folder=OUT/f'full_rally_{rally}'
    cap=cv2.VideoCapture(str(folder/'bytetrack.mp4'))
    assert cap.isOpened()
    first=frames[0]
    for frame in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame-first)
        ok,image=cap.read()
        assert ok,(rally,frame)
        cv2.imwrite(str(OUT/f'full_rally_{rally}_F{frame}.jpg'),image)
    cap.release()
