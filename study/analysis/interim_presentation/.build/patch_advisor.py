from pathlib import Path
import zipfile,copy,re,json
from lxml import etree as E

B=Path(__file__).parent
SRC=Path('C:/Users/ta239/Desktop/中山研/23238007R_MID.pptx')
NS={'p':'http://schemas.openxmlformats.org/presentationml/2006/main','a':'http://schemas.openxmlformats.org/drawingml/2006/main','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
P,A,R=NS['p'],NS['a'],NS['r'];PKG='http://schemas.openxmlformats.org/package/2006/relationships'
with zipfile.ZipFile(SRC) as z: data={n:z.read(n) for n in z.namelist()}
with zipfile.ZipFile(B/'advisor_donor.pptx') as z: donor={n:z.read(n) for n in z.namelist()}
def parse(b):return E.fromstring(b)
def save(root):return E.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
def texts(root):return ''.join(root.xpath('.//a:t/text()',namespaces=NS))
def settext(container,value,size=None):
 ps=container.findall('a:p',NS)
 if not ps:return
 props=ps[0].find('a:pPr',NS);rp=container.find('.//a:rPr',NS)
 for old in ps:container.remove(old)
 for line in value.split('\n'):
  p=E.SubElement(container,f'{{{A}}}p')
  if props is not None:p.append(copy.deepcopy(props))
  r=E.SubElement(p,f'{{{A}}}r');rpr=copy.deepcopy(rp) if rp is not None else E.Element(f'{{{A}}}rPr')
  if size:rpr.set('sz',str(size))
  r.append(rpr);E.SubElement(r,f'{{{A}}}t').text=line
def replace(root,old,new,size=None):
 for c in root.xpath('.//p:txBody|.//a:txBody',namespaces=NS):
  if texts(c)==old:settext(c,new,size);return True
 return False

# Native Artifact Tool slide bodies are spliced into the original package.
# This keeps the original animation timing, media, masters and slide IDs intact.
for sid,did in [(21,1),(6,2),(14,3),(15,4),(16,5),(17,6)]:
 root=parse(data['ppt/slides/slide6.xml'] if sid==21 else data[f'ppt/slides/slide{sid}.xml'])
 cs=root.find('p:cSld',NS);idx=list(root).index(cs);root.remove(cs)
 root.insert(idx,copy.deepcopy(parse(donor[f'ppt/slides/slide{did}.xml']).find('p:cSld',NS)))
 for child in list(root):
  if E.QName(child).localname in ['timing','transition']:root.remove(child)
 data[f'ppt/slides/slide{sid}.xml']=save(root)

order=list(range(1,6))+[21]+list(range(6,21))
pres=parse(data['ppt/presentation.xml']);lst=pres.find('p:sldIdLst',NS)
rels=parse(data['ppt/_rels/presentation.xml.rels']);used={r.get('Id') for r in rels};n=100
while f'rId{n}' in used:n+=1
rid=f'rId{n}';E.SubElement(rels,f'{{{PKG}}}Relationship',Id=rid,Type=R+'/slide',Target='slides/slide21.xml')
lst.insert(5,E.Element(f'{{{P}}}sldId',id=str(max(int(s.get('id')) for s in lst)+1),attrib={f'{{{R}}}id':rid}))
data['ppt/presentation.xml']=save(pres);data['ppt/_rels/presentation.xml.rels']=save(rels)
rel=parse(data['ppt/slides/_rels/slide6.xml.rels'])
for r in list(rel):
 if r.get('Type').endswith('/notesSlide'):r.set('Target','../notesSlides/notesSlide21.xml')
 elif not r.get('Type').endswith('/slideLayout'):rel.remove(r)
data['ppt/slides/_rels/slide21.xml.rels']=save(rel)
data['ppt/notesSlides/notesSlide21.xml']=data['ppt/notesSlides/notesSlide6.xml']
rel=parse(data['ppt/notesSlides/_rels/notesSlide6.xml.rels'])
for r in rel:
 if r.get('Type').endswith('/slide'):r.set('Target','../slides/slide21.xml')
data['ppt/notesSlides/_rels/notesSlide21.xml.rels']=save(rel)
ct=parse(data['[Content_Types].xml']);CT=E.QName(ct).namespace
for folder,part,kind in [('slides','slide','slide'),('notesSlides','notesSlide','notesSlide')]:
 E.SubElement(ct,f'{{{CT}}}Override',PartName=f'/ppt/{folder}/{part}21.xml',ContentType=f'application/vnd.openxmlformats-officedocument.presentationml.{kind}+xml')
data['[Content_Types].xml']=save(ct)

notes={
1:'ソフトテニスの試合映像を用いた動作予測手法の提案について，朝隈が発表します。今回は，前衛の予測に関係する動作の分析結果を報告します。',
2:'ダブルスでは，ネット付近の前衛と，後方で打ち合う後衛がペアを組みます。前衛が横へ出て，相手の球を途中で捉えるプレーがポーチです。今回は，この前衛が動くタイミングに着目します。',
3:'早く動きすぎると，相手後衛に動きを見られ，空いたコースへ打たれる可能性があります。この図は，その状況を示した模式例です。',
4:'一方，相手の打球を確認してから動こうとすると，動き出しが遅れ，ボールに届かない可能性があります。',
5:'そこで，相手に読まれにくく，球にも届くタイミングを知りたいと考えました。対戦相手による違いも調べ，選手や指導者が，相手に応じた出方を検討できるようにすることが最終目標です。',
21:'本研究で関心があるのは，前衛による相手の打球方向の予測です。今回はその手掛かりとして，打球前の動きを分析します。入力は前衛の位置系列と，相手の打球時刻・方向です。移動方向と打球方向を照合し，時点別・選手別の一致率を出力します。将来の動作を出力する予測モデルは，まだ実装していません。',
6:'先行研究では，前衛の動きを種類ごとに分類し，出現傾向を調べています。本研究では，打球前の各時点で，移動方向と打球方向がどの程度対応するかを数値化します。分析の観点の違いは，打球前の時点別に対応を調べることです。今後，プレーの成否も加え，選手の出方の検討や具体的な指導に役立てることを目指します。',
7:'対象は，2025年全日本選手権の男子決勝1試合です。相手後衛の1回の打球を1試技として，98試技を整理しました。前衛は2人で，P1が42試技，P2が56試技です。チェンジコート後も，同じ選手番号で集計しています。',
8:'前衛の位置の抽出には，姿勢推定モデルのYOLO11m-poseを使用しました。姿勢推定は，画像から足首などの身体の点を求める処理です。黄枠が対象前衛，紫の丸が足首です。追跡対象の取り違えは人手で確認・修正しました。',
9:'ボールがラケットに当たる画像を人手で確定し，打球時刻としました。打球方向は，対象前衛本人から見た左右で記録しています。足首位置は射影変換でコート上の座標に変換し，手前と奥の見え方をそろえます。',
10:'打球時刻を0秒とし，各時点の直前0.1秒の横方向の移動を調べます。例えば0.3秒前では，0.4秒前からの位置の変化を使います。コート幅の1.5％以上動いた場合に左右を判定し，それ未満は一致率の集計から外します。初動時刻の検出ではなく，区間ごとの方向判定です。',
11:'全体の一致率は，0.5秒前が42.9％，0.3秒前が64.0％，0.2秒前が73.3％，0.1秒前が68.2％でした。0.2秒前まで上がりましたが，直前には下がっています。ただし，これだけで予測能力の変化とは判断できません。',
12:'選手別では，0.1秒前の一致率がP1は81.8％，P2は54.5％でした。今回の試技では選手間の違いが見られます。ただし，両者ともこの時点の判定数は11件です。能力差と結論せず，試技を増やして確認します。',
13:'移動量が閾値を満たした試技は，全体の14.3％から25.5％でした。一致率は，この条件を満たした試技についての値であり，98試技すべてに対する割合ではありません。',
14:'時点ごとに集計対象が異なります。0.2秒前の15試技と，0.1秒前の22試技で，共通するのは11試技です。そのため，一致率の低下だけで，打球直前に予測が悪くなったとは言えません。また，不一致にはフェイントや戻りの可能性もあります。',
15:'今後は複数試合を集め，同じ試技にそろえた比較を行います。さらに，ポーチの開始時刻と成功・失敗を記録します。選手や対戦相手，開始位置の違いを整理し，タイミングと成否の関係を調べます。',
16:'今回は，打球前の移動方向と打球方向の一致率を，時点別・選手別に示しました。卒論では，この相手にはいつ出ると成功しやすいかを，映像と数値で検討できるようにすることを目指します。',
17:'参考文献です。以上で発表を終わります。ご清聴ありがとうございました。'
}
source_url='https://kanazawa-u.repo.nii.ac.jp/record/40387/files/AN0004423X-24-2-141-156.pdf'
notes[5]+='今回の分析では，まず打球前の動きと実際の打球方向との対応を確認します。'
notes[7]+='1ラリーに複数の打球があれば，それぞれを別の試技として扱います。'
notes[9]+='打球方向と移動方向は同じ選手視点にそろえ，画面上の左右とは区別します。'
notes[11]+='この表の分母は各時点で左右を判定した試技数です。例えば0.2秒前は，15試技のうち11試技が一致しました。'
notes[12]+='全体集計に加えて，個々の選手の傾向を確認するための結果です。'
old_total=0
for sid in range(1,18):
 root=parse(data[f'ppt/notesSlides/notesSlide{sid}.xml']);
 for sp in root.findall('.//p:sp',NS):
  ph=sp.find('.//p:ph',NS)
  if ph is not None and ph.get('type')=='body':old_total+=len(re.sub(r'\s','',texts(sp)))

for page,sid in enumerate(order,1):
 key=f'ppt/slides/slide{sid}.xml';root=parse(data[key])
 # Explicit footer numbers on native text objects, including multi-run digits.
 if sid!=1:
  for sp in root.findall('.//p:sp',NS):
   if texts(sp).strip()==str(sid):
    tx=sp.find('p:txBody',NS)
    if tx is not None:settext(tx,str(page))
 if sid==7:
  for sp in root.findall('.//p:sp',NS):
   if texts(sp).startswith('今期の実施内容'):settext(sp.find('p:txBody',NS),'今回の対象：1試合・98試技',3150)
 if sid==12:
  for sp in root.findall('.//p:sp',NS):
   if texts(sp).startswith('−0.1秒の推移'):settext(sp.find('p:txBody',NS),'−0.1秒では，選手間で一致率が異なる',2700)
 if sid==14:pass
 if sid in [3,4,5]:
  # Animation XML itself is retained byte-for-byte in meaning.
  pass
 data[key]=save(root)
 if sid in notes:
  nk=f'ppt/notesSlides/notesSlide{sid}.xml';nr=parse(data[nk])
  for sp in nr.findall('.//p:sp',NS):
   ph=sp.find('.//p:ph',NS)
   if ph is not None and ph.get('type')=='body':
    value=notes[sid]
    if sid==6:value+='\n\n出典・補足（読み上げない）：井箟ほか（1987），p.153「まとめ」。検索エンジン収録の本文で種類・出現傾向の分析と，動きの成果等を未検討とする記述を確認。比較はこの論文との分析観点の違いであり，全先行研究に対する未実施の証明ではない。\n'+source_url
    settext(sp.find('p:txBody',NS),value)
  data[nk]=save(nr)

# Strict preservation checks for the user's native animations and result tables.
with zipfile.ZipFile(SRC) as z:
 for sid in [3,4,5]:
  before=parse(z.read(f'ppt/slides/slide{sid}.xml')).find('p:timing',NS)
  after=parse(data[f'ppt/slides/slide{sid}.xml']).find('p:timing',NS)
  assert E.tostring(before,method='c14n')==E.tostring(after,method='c14n')
 for sid in [7,10,11,12,13,18]:
  before=parse(z.read(f'ppt/slides/slide{sid}.xml')).findall('.//a:tbl',NS)
  after=parse(data[f'ppt/slides/slide{sid}.xml']).findall('.//a:tbl',NS)
  assert [texts(t) for t in before]==[texts(t) for t in after]
new_total=sum(len(re.sub(r'\s','',x)) for x in notes.values())
metrics={'old_note_chars':old_total,'new_script_chars':new_total,'reduction_percent':round(100*(1-new_total/old_total),1),'slide_count':21,'main_plus_references':18,'source':str(SRC)}
(B/'advisor_metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
assert new_total<=old_total*.82,(new_total,old_total)
out=B.parent/'output';out.mkdir(exist_ok=True)
md=['# 先生指摘対応版 発表原稿','本編17枚＋参考文献1枚。最後の3枚は質疑用。','']
for page,sid in enumerate(order,1):
 if sid in notes:
  title=texts(parse(data[f'ppt/slides/slide{sid}.xml']).find('.//p:sp',NS));md += [f'## {page}．{title}',notes[sid],'']
(out/'発表原稿_先生指摘対応_v11.md').write_text('\n\n'.join(md),encoding='utf-8')
with zipfile.ZipFile(B/'advisor_candidate.pptx','w',zipfile.ZIP_DEFLATED) as z:
 for k,v in data.items():z.writestr(k,v)
print(json.dumps(metrics,ensure_ascii=False))
