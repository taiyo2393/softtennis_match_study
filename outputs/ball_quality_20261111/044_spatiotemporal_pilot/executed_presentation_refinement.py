"""Two presentation-only label moves; keep frozen numerical/source code intact."""
import build_044_spatiotemporal_pilot as pilot
from PIL import ImageDraw

original=ImageDraw.ImageDraw.text
def positioned(self,xy,text,*args,**kwargs):
    # Candidate labels need their own row, clear of the explanatory sentence.
    if len(text)==3 and text.startswith('C') and text[1:].isdigit() and xy[1]==908:
        xy=(xy[0],935)
    # Keep the original scoreboard unobstructed; show active candidate below video.
    if xy==(32,115) and text.startswith('方向変化候補 '):
        xy=(780,836);text=text.split('（')[0]
    return original(self,xy,text,*args,**kwargs)

if __name__=='__main__':
    ImageDraw.ImageDraw.text=positioned
    try:pilot.render()
    finally:ImageDraw.ImageDraw.text=original
    pilot.shutil.copy2(__file__,pilot.OUT/'executed_presentation_refinement.py')
    pilot.dump(pilot.OUT/'presentation_refinement.json',dict(
        changes=['MoveC02/C04 labels off explanatory sentence onto the same row asC01/C03',
                 'Move active-candidate text away from source scoreboard to below-video status row'],
        numerical_data_or_event_candidates_changed=False,source_images_changed=False))
