"""Create raw chronological contact sheets for the next two points."""
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont

import build_044_spatiotemporal_pilot as common

BASE = common.HERE / 'match2_ball_first_point'
OUT = common.HERE / 'match2_points_2_3'
SOURCE = Path(common.load(BASE / 'source_manifest.json')['registration']['source_video'])


def cache_point(point, first, last):
    import prepare_match2_ball_first_point as raw_review

    folder = OUT / f'point_{point:02}'
    frames = folder / 'frames'
    frames.mkdir(parents=True, exist_ok=False)
    cap = cv2.VideoCapture(str(SOURCE))
    try:
        assert cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        for frame in range(first, last+1):
            ok, image = cap.read()
            assert ok and image.shape == (1080,1920,3), frame
            assert cv2.imwrite(str(frames/f'{frame}.jpg'), image, [cv2.IMWRITE_JPEG_QUALITY,95])
    finally:
        cap.release()
    raw_review.OUT = folder
    raw_review.sheet('score_sequence.jpg', [first, first+60, last-60, last], (0,0,640,150), 400, 2)
    if point == 2:
        raw_review.sheet('serve_sequence.jpg', range(13700,13801,5), (700,435,1230,650), 340, 5)
        raw_review.sheet('end_sequence.jpg', range(13980,14141,8), (240,425,1660,930), 340, 5)
    else:
        raw_review.sheet('serve_sequence.jpg', range(15210,15321,5), (700,435,1230,650), 340, 5)
        raw_review.sheet('end_sequence.jpg', range(15435,15572,8), (240,425,1660,930), 340, 5)


def timeline(first, last, step, name):
    frames = list(range(first, last + 1, step))
    width, height, columns = 320, 206, 5
    sheet = Image.new('RGB', (width*columns, height*((len(frames)+columns-1)//columns)), '#101820')
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(str(common.FONT), 16)
    cap = cv2.VideoCapture(str(SOURCE))
    try:
        assert cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        for frame in range(first, last+1):
            if (frame-first)%step:
                assert cap.grab(), frame
                continue
            ok, image = cap.read()
            assert ok and image.shape == (1080,1920,3), frame
            rgb = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
            i = (frame-first)//step
            x, y = i%columns*width, i//columns*height
            sheet.paste(rgb.resize((width,180)), (x,y+26))
            sheet.paste(rgb.crop((0,0,548,145)).resize((210,56)), (x,y+26))
            draw.text((x+6,y+3), f'F{frame}   {frame/common.FPS:.2f}s', fill='white', font=font)
    finally:
        cap.release()
    OUT.mkdir(exist_ok=True)
    sheet.save(OUT/name, quality=95)


if __name__ == '__main__':
    cache_point(2,13547,14147)
    cache_point(3,15107,15587)
