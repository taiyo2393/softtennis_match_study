"""Review only four newly proposed weak components, without treating them as truth."""
from pathlib import Path
import subprocess,shutil
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
import build_044_spatiotemporal_pilot as b
from prepare_056_route_transfer import FRAMES,OFFSET
from trial_056_temporal_reacquisition import OUT,FIRST,LAST

GREEN=(80,230,150);BLUE=(70,190,255)

def main():
    assert not (OUT/'056_four_reacquisition_candidates.mp4').exists()
    rows=b.read(OUT/'proposed_path.csv');by={int(r['source_frame']):r for r in rows}
    new=[r for r in rows if r['kind']=='weak_heatmap_candidate_unconfirmed']
    fonts={s:ImageFont.truetype(str(b.FONT),s) for s in [20,23,30]}
    fps=b.FPS/4
    proc=subprocess.Popen(['C:/ffmpeg/bin/ffmpeg.exe','-y','-loglevel','error','-f','rawvideo',
        '-pix_fmt','rgb24','-s','1920x720','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264',
        '-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'056_four_reacquisition_candidates.mp4')],
        stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    mapping=[];sheet=Image.new('RGB',(1280,800),(12,22,32));sheet_draw=ImageDraw.Draw(sheet)
    number=0
    for f in range(FIRST,LAST+1):
        r=by[f];rgb=Image.open(FRAMES/f'{f-OFFSET:06d}.jpg').convert('RGB')
        frame=Image.new('RGB',(1920,720),(12,22,32));d=ImageDraw.Draw(frame)
        d.text((20,10),'056：欠測から拾い直した4候補（青は未確認）',font=fonts[30],fill='white')
        d.text((20,57),f'F{f} ｜ 4分の1速＋新候補で短く停止 ｜ 左：全体　右：固定範囲を拡大',font=fonts[20],fill='white')
        frame.paste(rgb.resize((960,540)),(0,100))
        crop=rgb.crop((630,440,1142,728))
        frame.paste(crop.resize((960,540)),(960,100));d=ImageDraw.Draw(frame)
        is_new=r['kind']=='weak_heatmap_candidate_unconfirmed'
        if r['proposed_x_px']!='':
            x,y=float(r['proposed_x_px']),float(r['proposed_y_px']);color=BLUE if is_new else GREEN
            for xx,yy in [(x/2,100+y/2),(960+(x-630)*1.875,100+(y-440)*1.875)]:
                if is_new:d.line([(xx,yy-10),(xx+10,yy),(xx,yy+10),(xx-10,yy),(xx,yy-10)],fill=color,width=3)
                else:d.ellipse((xx-8,yy-8,xx+8,yy+8),outline=color,width=3)
            if is_new:
                tile=crop.resize((640,360));td=ImageDraw.Draw(tile)
                xx=(x-630)*1.25;yy=(y-440)*1.25
                td.ellipse((xx-10,yy-10,xx+10,yy+10),outline=BLUE,width=3)
                tx=(number%2)*640;ty=(number//2)*400
                sheet.paste(tile,(tx,ty+40));sheet_draw=ImageDraw.Draw(sheet)
                sheet_draw.text((tx+12,ty+6),f'{chr(65+number)} : F{f}',font=fonts[23],fill=BLUE)
                number+=1
        d.text((20,651),'緑○：前回の区間確認あり　青◇：今回の新候補（未確認）',font=fonts[23],fill=(230,240,245))
        d.text((20,686),'球候補のない時点は空白。青の正しさだけを確認し、見えない球の位置は決めません。',font=fonts[20],fill=(230,240,245))
        if is_new:frame.save(OUT/f'new_candidate_F{f}.jpg',quality=95)
        for repeat in range(15 if is_new else 1):
            proc.stdin.write(np.asarray(frame).tobytes())
            mapping.append(dict(display_frame=len(mapping),source_frame=f,new_candidate=int(is_new),hold_repeat=repeat))
    proc.stdin.close();err=proc.stderr.read();assert proc.wait()==0,err
    assert number==4
    sheet.save(OUT/'four_new_candidates.jpg',quality=95)
    b.write(OUT/'video_source_frame_map.csv',mapping)
    cap=cv2.VideoCapture(str(OUT/'056_four_reacquisition_candidates.mp4'));count=0
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape==(720,1920,3)
        if count==15:cv2.imwrite(str(OUT.parent/'056_reacquisition_decoded_check.jpg'),im)
        count+=1
    cap.release();assert count==len(mapping)==84
    protocol=b.load(OUT/'protocol.json')
    assert all(b.sha(Path(p))==digest for p,digest in protocol['protected_input_hashes'].items())
    result=b.load(OUT/'result.json')
    b.dump(OUT/'verification.json',dict(result,encoded_and_decoded_frames=count,
        source_frames=list(range(FIRST,LAST+1)),crop_native_xyxy=[630,440,1142,728],
        uniform_crop_enlargement=1.875,source_aspect_preserved=True,old_publications_unchanged=True,
        newly_proposed_frames=[int(r['source_frame']) for r in new],review_pending=True))
    (OUT/'README.md').write_text(
        '# 056：前後の確認済み候補を使った欠測内の再探索試行\n\n'
        '元の重み・入力・AMP・バッチを再現し、28フレームの元出力を座標・信頼度まで完全再現した。'
        '確認済み15候補は全て保持。欠測13フレームだけを別の抽出方法で調べた。\n\n'
        '元の0.3閾値では残らない弱い成分を0.1閾値で抽出し、両端の確認済み位置を結んだ参考線の'
        '32px以内に中心を持つ成分が1個だけの場合に候補を出す。参考線の位置そのものは出力しない。'
        '0.1と32pxは今回一組だけ固定した探索設定であり、一般化性能は未検証。未来側の位置も使うオフライン処理。\n\n'
        '新候補はF154861・154862・154873・154878の4時点。残り9時点は未取得。'
        '新候補の正しさは未確認で、検出回復4件・精度改善という結論にはしない。'
        '球が本当に不可視か、同じ映像を使った別手法で取得可能かも未確定。\n\n'
        '動画の緑○は前回の区間確認のある元出力、青◇は今回の未確認候補。'
        '元データ・既存の確認・旧動画は変更していない。\n',encoding='utf-8')
    shutil.copy2(__file__,OUT/'executed_render.py')
    b.dump(OUT/'publication_hashes.json',{p.name:b.sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='publication_hashes.json'})
    print('verified',count,'encoded frames; new candidates:',[r['source_frame'] for r in new])

if __name__=='__main__':main()
