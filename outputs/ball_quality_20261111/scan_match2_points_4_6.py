"""Review chronological raw footage around the next three scoreboard changes."""
import sys

import cv2
from PIL import Image, ImageDraw, ImageFont

import build_044_spatiotemporal_pilot as common
from scan_match2_next_points import SOURCE


OUT = common.HERE / 'match2_points_4_6'


def make_sheet(first, last, step, name, *, court_view=True):
    OUT.mkdir(exist_ok=True)
    frames = list(range(first, last + 1, step))
    width, height, columns = (320, 210, 5) if court_view else (155, 88, 8)
    sheet = Image.new('RGB', (width * columns, height * ((len(frames) + columns - 1) // columns)), '#101820')
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(str(common.FONT), 15)
    capture = cv2.VideoCapture(str(SOURCE))
    try:
        assert capture.isOpened() and capture.set(cv2.CAP_PROP_POS_FRAMES, first)
        for i, frame in enumerate(frames):
            while round(capture.get(cv2.CAP_PROP_POS_FRAMES)) < frame:
                assert capture.grab()
            ok, raw = capture.read()
            assert ok and raw.shape[:2] == (1080, 1920)
            assert round(capture.get(cv2.CAP_PROP_POS_FRAMES)) == frame + 1
            image = Image.fromarray(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
            x, y = (i % columns) * width, (i // columns) * height
            if court_view:
                sheet.paste(image.resize((320, 180)), (x, y + 26))
                board = image.crop((448, 30, 581, 151)).resize((69, 62))
                sheet.paste(board, (x + 244, y + 29))
            else:
                sheet.paste(image.crop((448, 30, 581, 151)).resize((70, 64)), (x + 8, y + 24))
            draw.text((x + 6, y + 3), f'F{frame}', fill='white', font=font)
    finally:
        capture.release()
    sheet.save(OUT / name, quality=94)


if __name__ == '__main__':
    first, last, step = map(int, sys.argv[1:4])
    make_sheet(first, last, step, sys.argv[4], court_view='score' not in sys.argv[5:])
