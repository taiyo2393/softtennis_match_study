"""Extract the first point from the registered source for event review."""
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont

import build_044_spatiotemporal_pilot as common


PREVIOUS = common.HERE.parent / 'ball_quality_20261110'
RAW_REFERENCE = PREVIOUS / 'match2_camera_setup/first_point_raw'
OUT = common.HERE / 'match2_ball_first_point'
FIRST, LAST = 12228, 12647


def extract_frames():
    registration = common.load(PREVIOUS / 'match2_external_validation/source_registration.json')
    source = Path(registration['source_video'])
    assert source.stat().st_size == registration['bytes']
    frames = OUT / 'frames'
    frames.mkdir(parents=True, exist_ok=False)
    anchors = {12288: 205, 12408: 207, 12527: 209, 12647: 211}
    capture = cv2.VideoCapture(str(source))
    try:
        assert capture.set(cv2.CAP_PROP_POS_FRAMES, FIRST)
        for frame in range(FIRST, LAST + 1):
            ok, image = capture.read()
            assert ok and image.shape == (1080, 1920, 3), frame
            if frame in anchors:
                reference = cv2.imread(str(RAW_REFERENCE / f'raw_{anchors[frame]}s_F{frame}.png'))
                assert reference is not None and (image == reference).all(), frame
            assert cv2.imwrite(str(frames / f'{frame}.jpg'), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    finally:
        capture.release()
    common.dump(OUT / 'source_manifest.json', {
        'registration': registration,
        'source_frame_range_inclusive': [FIRST, LAST],
        'pixel_exact_match_to_existing_raw_references': list(anchors),
        'cached_frame_format': '1920x1080 JPEG quality95',
        'cached_image_sha256': {p.name: common.sha(p) for p in sorted(frames.iterdir())},
    })


def sheet(name, times, crop, width, columns):
    left, top, right, bottom = crop
    height = round((bottom - top) * width / (right - left))
    times = list(times)
    cell_height = height + 27
    image = Image.new('RGB', (columns * width, ((len(times) + columns - 1) // columns) * cell_height), '#101820')
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(common.FONT), 16)
    for index, frame in enumerate(times):
        x, y = index % columns * width, index // columns * cell_height
        with Image.open(OUT / 'frames' / f'{frame}.jpg') as source:
            image.paste(source.crop(crop).resize((width, height), Image.Resampling.LANCZOS), (x, y + 27))
        draw.text((x + 8, y + 3), f'F{frame}  {frame/common.FPS:.3f}s', font=font, fill='white')
    image.save(OUT / name, quality=94)


if __name__ == '__main__':
    extract_frames()
    sheet('serve_raw.jpg', range(12276, 12391, 6), (680, 420, 1120, 650), 330, 4)
    sheet('end_raw.jpg', range(12498, 12613, 6), (270, 415, 1660, 900), 425, 4)
