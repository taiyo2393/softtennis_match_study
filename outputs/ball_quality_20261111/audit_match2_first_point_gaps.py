"""Draw raw-frame sheets covering all six gaps in the frozen baseline."""
from PIL import Image, ImageDraw, ImageFont

import build_044_spatiotemporal_pilot as common


BASE = common.HERE / 'match2_ball_first_point'
OUT = common.HERE / 'match2_first_point_gap_audit'
SHEETS = [
    ('gap_01.jpg', 12386, 12391, (1040, 450, 1200, 560), 300, 3),
    ('gap_02_a.jpg', 12419, 12431, (1140, 440, 1480, 800), 285, 5),
    ('gap_02_b.jpg', 12432, 12443, (1140, 440, 1480, 800), 285, 5),
    ('gap_03_04.jpg', 12443, 12456, (1130, 430, 1450, 650), 295, 5),
    ('gap_05_06.jpg', 12460, 12473, (1080, 425, 1430, 610), 300, 5),
]


def main():
    OUT.mkdir(exist_ok=True)
    manifest = common.load(BASE / 'source_manifest.json')['cached_image_sha256']
    gaps = common.read(common.HERE / 'match2_first_point_ball_evaluated/baseline_candidate_gaps.csv')
    missing = {frame for gap in gaps for frame in range(int(gap['first_source_frame']), int(gap['last_source_frame'])+1)}
    covered = set()
    font = ImageFont.truetype(str(common.FONT), 15)
    for name, first, last, crop, width, columns in SHEETS:
        frames = range(first, last+1)
        covered.update(frames)
        height = round((crop[3]-crop[1])*width/(crop[2]-crop[0]))
        row_height = height+24
        sheet = Image.new('RGB', (width*columns, row_height*((len(frames)+columns-1)//columns)), '#101820')
        draw = ImageDraw.Draw(sheet)
        for index, frame in enumerate(frames):
            x, y = index%columns*width, index//columns*row_height
            path = BASE / 'frames' / f'{frame}.jpg'
            assert common.sha(path) == manifest[path.name]
            with Image.open(path) as image:
                sheet.paste(image.crop(crop).resize((width, height), Image.Resampling.LANCZOS), (x, y+24))
            draw.text((x+7, y+2), f'F{frame}', font=font, fill='white')
        sheet.save(OUT/name, quality=95)
    assert len(missing) == 44 and missing <= covered
    common.dump(OUT/'raw_sheet_manifest.json', dict(missing_frames=sorted(missing),
        raw_images_displayed=len(covered), sheet_specifications=SHEETS))


if __name__ == '__main__':
    main()
