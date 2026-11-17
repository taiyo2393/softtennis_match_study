import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {Presentation,PresentationFile,FileBlob} from '@oai/artifact-tool';

const buildDir=path.dirname(fileURLToPath(import.meta.url));
const workspaceDir=path.dirname(buildDir);
const projectDir=path.resolve(workspaceDir,'../..');
const outputDir=path.join(workspaceDir,'output');
const skillDir='C:/Users/ta239/.codex/plugins/cache/openai-primary-runtime/presentations/26.904.11930/skills/presentations';
const python='C:/Users/ta239/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
process.env.RUNTIME_NODE_MODULES='C:/Users/ta239/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
process.env.RUNTIME_NODE=process.execPath;
process.env.RUNTIME_PYTHON=python;
const {finalizePresentation,resolvePresentationFont}=await import(pathToFileURL(path.join(skillDir,'container_tools/artifact_tool_utils.mjs')).href);
const font=resolvePresentationFont({fontFamily:'Meiryo',availableFonts:['Meiryo']});
const revision=process.argv[2]??'v1';

function csvParse(text){
  const rows=[];let row=[],cell='',quoted=false;
  for(let i=0;i<text.length;i++){
    const c=text[i];
    if(c==='"'){if(quoted&&text[i+1]==='"'){cell+='"';i++;}else quoted=!quoted;}
    else if(c===','&&!quoted){row.push(cell);cell='';}
    else if(c==='\n'&&!quoted){row.push(cell.replace(/\r$/,''));if(row.some(Boolean))rows.push(row);row=[];cell='';}
    else cell+=c;
  }
  if(cell||row.length){row.push(cell.replace(/\r$/,''));rows.push(row);}
  const headers=rows.shift().map(x=>x.replace(/^\uFEFF/,''));
  return rows.map(a=>{assert.equal(a.length,headers.length);return Object.fromEntries(headers.map((h,i)=>[h,a[i]]));});
}
const eventPath=path.join(projectDir,'analysis/match1_court_plane_directional_motion_t0p015_event_timepoints.csv');
const identityPath=path.join(projectDir,'annotations/match1_target_front_identity_review.csv');
const pooledPath=path.join(projectDir,'analysis/match1_court_plane_directional_motion_t0p015_summary.csv');
const events=csvParse(await fs.readFile(eventPath,'utf8'));
const identities=new Map(csvParse(await fs.readFile(identityPath,'utf8')).map(x=>[x.event_id,x]));
const pooled=csvParse(await fs.readFile(pooledPath,'utf8'));
assert.equal(events.length,392);assert.equal(new Set(events.map(x=>x.event_id)).size,98);
assert.equal(new Set(events.map(x=>x.event_id+'@'+x.timepoint_sec)).size,392);
for(const e of events){
  const id=identities.get(e.event_id);assert.equal(id.identity_status,'confirmed');
  for(const key of ['target_front_player_id','target_front_camera_end','impact_source_frame'])assert.equal(e[key],id[key]);
  const d=Number(e.lateral_displacement_court_width);
  const positive=e.target_front_camera_end==='near'?'front_right':'front_left';
  const negative=positive==='front_right'?'front_left':'front_right';
  const direction=d>=.015?positive:d<=-.015?negative:'stay';
  assert.equal(direction,e.movement_direction_target_front_view);
  assert.equal(e.direction_matches_actual,direction==='stay'?'':String(Number(direction===e.actual_direction_target_front_view)));
}
const times=[-.5,-.3,-.2,-.1];
const data=times.map(t=>['P1','P2','ALL'].map(player=>{
  const rows=events.filter(e=>Number(e.timepoint_sec)===t&&(player==='ALL'||e.target_front_player_id===player));
  const n=rows.length,d=rows.filter(e=>e.movement_is_directional==='1').length,k=rows.filter(e=>e.direction_matches_actual==='1').length;
  return {player,t,n,d,k,agreement:k/d,coverage:d/n};
}));
for(const [a,b,c] of data){
  assert.equal(a.n,42);assert.equal(b.n,56);assert.equal(c.n,98);
  for(const key of ['n','d','k'])assert.equal(a[key]+b[key],c[key]);
  const q=pooled.find(x=>Number(x.timepoint_sec)===c.t);
  assert.equal(c.d,Number(q.directional_events));assert.equal(c.k,Number(q.matching_direction_events));
  assert.equal(c.agreement.toFixed(3),q.conditional_match_rate);
}
await fs.writeFile(path.join(buildDir,'verified_data.json'),JSON.stringify(data,null,2));

const p=Presentation.create({slideSize:{width:960,height:720}});
const C={ink:'#182126',blue:'#214F76',muted:'#54616A',gray:'#F3F5F6',border:'#AAB7C0'};
const noteEntries=[];
const tableOwners=[];
const methodsSource='提出予稿：C:/Users/ta239/Desktop/中山研/23238007R_MID_04.doc\n解析：'+eventPath+'\n選手ラベル：'+identityPath;

function txt(s,value,x,y,w,h,size=30,bold=false,color=C.ink,align='left'){
  const sh=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  sh.text=value;sh.text.style={typeface:font,fontSize:size,bold,color,autoFit:'none',alignment:align,verticalAlignment:'middle'};return sh;
}
function base(title,number,backup=false){
  const s=p.slides.add();s.background.fill='#FFFFFF';
  txt(s,title,50,30,860,76,42,true,C.blue);
  txt(s,backup?'補足 '+(number-10):String(number),760,674,150,30,20,false,C.muted,'right');
  return s;
}
function notes(s,title,sec,script,extra=''){
  const body=(sec?'目安 '+sec+'秒\n\n':'質疑用\n\n')+script+'\n\n出典・定義\n'+methodsSource+(extra?'\n'+extra:'');
  s.speakerNotes.textFrame.setText(body);noteEntries.push({title,sec,script,extra});
}
function imageSlot(s,label,detail,x,y,w,h){
  const sh=s.shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:h},fill:C.gray,line:{fill:C.border,width:1.5}});
  txt(s,label,x+18,y+h/2-58,w-36,44,27,true,C.blue,'center');
  txt(s,detail,x+20,y+h/2-9,w-40,92,23,false,C.muted,'center');
  return sh;
}
function table(s,values,{x=55,y=156,w=850,rowH=83,widths=[175,225,225,225],body=30,header=27}={}){
  const t=s.tables.add({rows:values.length,columns:values[0].length,left:x,top:y,width:w,height:rowH*values.length,columnWidths:widths,values});
  t.styleOptions={headerRow:false,bandedRows:false,firstColumn:false,lastColumn:false};
  t.borders.assign({style:'solid',fill:'#CAD1D6',width:1});
  for(let r=0;r<values.length;r++){
    t.rows[r].height=rowH;
    for(let c=0;c<values[0].length;c++){
      const cell=t.getCell(r,c);cell.fill=r===0?'#EAF0F4':'#FFFFFF';
      cell.text.style={typeface:font,fontSize:r===0?header:body,bold:r===0,color:C.ink,alignment:'center',verticalAlignment:'middle',autoFit:'none'};
    }
  }
  tableOwners.push(p.slides.items.length);return t;
}
const pct=v=>(v*100).toFixed(1)+'％';
const timeLabel=t=>'−'+Math.abs(t).toFixed(1);

// 1. Cover
{
const s=p.slides.add();s.background.fill='#FFFFFF';
txt(s,'ソフトテニスの試合映像を用いた\n動作予測手法の提案',65,155,830,160,47,true,C.blue,'center');
txt(s,'23238007　朝隈 大耀',70,414,820,55,32,false,C.ink,'center');
txt(s,'佐賀大学 理工学部 理工学科\n知能情報システム工学コース',70,480,820,78,25,false,C.muted,'center');
txt(s,'指導教員　中山 功一 教授',70,587,820,45,27,false,C.ink,'center');
notes(s,'題目',15,'ソフトテニスの試合映像を用いた動作予測手法の提案について，朝隈が発表します．今回は，前衛の移動方向と相手後衛の打球方向が，打球前の各時点でどの程度一致するかを調べました．');
}
// 2. Background / image supplied by user
{
const s=base('ソフトテニスの前衛と後衛',2);
imageSlot(s,'画像①　コート全景','4人が写り，前衛と後衛の\n位置関係が分かる試合画像',50,145,560,315);
txt(s,'ダブルスは2人1組',635,146,280,55,28,true);
txt(s,'前衛\nネット付近で\n相手の打球を捉える',635,225,280,125,27);
txt(s,'後衛\nコート後方を中心に\n相手と打ち合う',635,365,280,125,27);
txt(s,'ポーチ：前衛が横へ動き，\n相手の打球を途中で捉えるプレー',55,533,850,104,32,true,C.blue);
notes(s,'ソフトテニスの前衛と後衛',40,'ソフトテニスのダブルスは2人1組で行います．代表的な陣形では，ネット近くの前衛と，コート後方の後衛に分かれます．後衛が打ち合う中で，前衛が横に動いて相手の球を途中で捉えるプレーをポーチと呼びます．前衛には，打球が来る方向を見極めながら，動き出すタイミングを選ぶことが求められます．今回着目したのは，この打球前の前衛の動きです．','画像①はユーザー提供予定．前衛・後衛は代表的な陣形での役割の説明であり，全場面で位置が固定されるわけではない．');
}
// 3. Purpose and usefulness
{
const s=base('研究目的と想定する活用',3);
txt(s,'前衛は，いつから打球方向に\n対応した動きを見せるのか',55,138,850,115,38,true,C.blue);
txt(s,'打球前の各時点で，前衛の移動方向と\n相手後衛の打球方向の一致率を求める．',55,277,850,94,32);
txt(s,'選手・指導者の振り返り',55,427,850,48,29,true);
txt(s,'自分の動きと打球方向が対応する時点を確認する．',55,480,850,52,27);
txt(s,'対戦相手に応じた準備',55,557,850,48,29,true);
txt(s,'前衛・ペアごとの動きの傾向を戦術検討に生かす．',55,610,850,45,27);
notes(s,'研究目的と想定する活用',45,'知りたいのは，前衛がどの時点から相手の打球方向に対応した動きをしているのか，ということです．そこで，打球前の各時点で前衛の移動方向と実際の打球方向を比較し，一致率を求めます．選手や指導者が，感覚だけでなく時刻と数値を使ってプレーを振り返ることを想定しています．さらに，選手や対戦ペアごとに集計すれば，前衛は自分が動くタイミングを，後衛は相手前衛の動きへの対応を検討する材料になります．');
}
// 4. Data
{
const s=base('分析対象',4);
txt(s,'2025年 全日本ソフトテニス選手権大会\n男子決勝の試合映像',55,136,850,99,33,true);
txt(s,'コート後方から撮影した1試合',55,240,850,53,28,false,C.muted);
txt(s,'98',65,329,310,133,100,true,C.blue);
txt(s,'試技',276,384,150,62,35,true);
txt(s,'相手後衛の1回の打球を1試技とする',55,475,850,52,29);
txt(s,'対象前衛 P1：42試技\n対象前衛 P2：56試技',488,326,417,135,35,true);
txt(s,'選手IDを保ち，コートの手前・奥を統合して集計',55,569,850,85,29);
notes(s,'分析対象',40,'対象は2025年の全日本選手権男子決勝，1試合です．コート後方からの映像を使い，相手後衛の1回の打球を1試技として，確認済みの98試技を分析しました．対象の前衛は2人で，P1が42試技，P2が56試技です．チェンジコートで映像上の手前と奥が入れ替わっても，同じ選手には同じIDを付けています．位置の見え方の違いを補正した上で，全体と選手別に集計しました．','P1/P2は確認済みの内部選手ID．実名への対応を推測しない．');
}
// 5. Methods
{
const s=base('分析手順',5);
txt(s,'1　打球時刻と方向を確認',55,140,850,49,30,true);
txt(s,'インパクトと実際の左右方向を人手で確定',85,198,820,47,26);
txt(s,'2　前衛の足首位置を抽出',55,284,405,83,29,true);
txt(s,'YOLO11m-pose［1］で推定\n追跡結果を人手で確認',85,373,370,91,26);
imageSlot(s,'画像②　前衛の追跡結果','対象前衛と両足首の点が\n分かる解析画面',480,284,425,239);
txt(s,'3　コート上の位置へ変換',55,547,850,49,30,true);
txt(s,'射影変換［2］後の両足首の中点を前衛位置とする',85,603,820,48,26);
notes(s,'分析手順',45,'分析手順です．まず，ラケットにボールが当たるインパクト時刻と，実際の打球方向を人手で確認しました．次に，姿勢推定モデルのYOLO11m-poseで対象前衛の足首位置を抽出し，追跡結果を確認しました．映像では奥の選手ほど小さく見えるため，射影変換を使ってコート上の座標に直します．今回は左右の足首をそれぞれ変換した後，その中点を前衛の位置としました．この位置の変化から移動方向を求めます．','［1］Ultralytics YOLO11: https://github.com/ultralytics/ultralytics\n［2］Hartley and Zisserman, Multiple View Geometry in Computer Vision, 2nd ed., 2004.\n画像②はユーザー提供予定．');
}
// 6. Operational definition
{
const s=base('移動方向の判定',6);
txt(s,'インパクトを0秒とし，打球前の4時点を評価',55,135,850,47,29);
txt(s,'−0.5秒　−0.3秒　−0.2秒　−0.1秒',55,192,850,47,32,true,C.blue);
txt(s,'例：−0.2秒の判定に使う2時点',55,265,850,39,26,true);
imageSlot(s,'画像③-A　−0.3秒','同じ試技の前衛位置\n（区間の始点）',55,313,400,225);
imageSlot(s,'画像③-B　−0.2秒','同じ試技の前衛位置\n（0.1秒後）',505,313,400,225);
txt(s,'直前0.1秒で，横にコート幅の1.5％以上移動',55,565,850,46,29,true);
txt(s,'左右は対象前衛本人の視点に統一',55,618,850,38,27);
notes(s,'移動方向の判定',45,'インパクトを0秒として，0.5秒前，0.3秒前，0.2秒前，0.1秒前の4時点を評価しました．各時点では，その直前0.1秒間の横方向の移動を見ます．例えば0.2秒前の判定には，0.3秒前と0.2秒前の位置の差を使います．コート幅の1.5パーセント以上移動した場合に，左または右と判定します．左右は対象前衛本人から見た方向に統一しました．それより小さな移動は左右の判定対象に含めません．','判定式：dx(t)=x(t)-x(t-0.1秒)．横座標はコート幅で正規化．閾値はabs(dx)>=0.015．画像③は同じ試技の2フレームを使用する．この定義は初動開始時刻の検出ではない．');
}
// 7. Pooled results
{
const s=base('結果：全体の方向の一致率',7);
txt(s,'一致率 ＝ 一致数 ÷ 左右を判定できた試技数',55,120,850,42,27,true);
const values=[['時点\n（秒）','判定試技数','一致数','一致率'],...data.map(([a,b,q])=>[timeLabel(q.t),String(q.d),String(q.k),pct(q.agreement)])];
table(s,values,{y:178,rowH:79,widths:[175,225,200,250],body:33,header:27});
txt(s,'全98試技．各時点で左右を判定できた試技のみ集計．',55,596,850,51,25);
notes(s,'結果：全体の方向の一致率',45,'こちらが全体の結果です．一致率は，移動方向と打球方向が一致した数を，左右を判定できた試技数で割った値です．0.5秒前は14試技中6試技で42.9パーセントでした．0.3秒前は64.0パーセント，0.2秒前は73.3パーセントです．一方，0.1秒前では68.2パーセントとなりました．0.5秒前から0.2秒前までは高くなっていますが，インパクトに近づくほど必ず上がる結果ではありません．','全体集計との照合元：'+pooledPath+'\n一致率の分母は98ではない．時点ごとに判定対象が異なる．');
}
// 8. Per-player results
{
const s=base('結果：選手別の方向の一致率',8);
txt(s,'一致率（一致数／判定試技数）',55,116,850,40,27);
const values=[['時点\n（秒）','P1\n全42試技','P2\n全56試技','全体\n全98試技'],...data.map(row=>[timeLabel(row[0].t),...row.map(q=>`${pct(q.agreement)}\n(${q.k} / ${q.d})`)])];
table(s,values,{y:165,rowH:88});
txt(s,'同じ選手の手前・奥の場面をまとめて集計',55,623,850,37,25);
notes(s,'結果：選手別の方向の一致率',45,'選手ごとに分けると，推移に違いが見られます．P1は50.0，66.7，71.4，81.8パーセントと，今回の4時点ではインパクトに近づくにつれて高くなりました．一方，P2は0.2秒前が75.0パーセントですが，0.1秒前では54.5パーセントに下がっています．全体だけの集計では見えにくい違いを，選手ごとに示せました．ただし分母が少ないため，この結果だけで選手の能力の優劣を判断するものではありません．');
}
// 9. Discussion
{
const s=base('考察',9);
txt(s,'打球方向と対応する動きの時間的な変化',55,140,850,54,33,true,C.blue);
txt(s,'全体では−0.5秒から−0.2秒にかけて一致率が上昇．\n−0.1秒では低下し，単調な上昇にはならなかった．',55,212,850,108,28);
txt(s,'選手別に異なる推移',55,367,850,51,33,true,C.blue);
txt(s,'選手・ペアごとの傾向を検討する材料になる．\nより多くの試技で，再現する違いかを確かめる．',55,434,850,100,28);
txt(s,'左右を判定できたのは全試技の14.3〜25.5％．\n一致率は「予測できていた割合」とは区別する．',55,579,850,80,25,false,C.muted);
notes(s,'考察',45,'全体では，打球に近い時点で方向が対応しやすくなる可能性が見られましたが，0.1秒前には低下しており，一定の傾向とは言い切れません．また，選手別の推移は異なるため，選手や相手ペアを分けた分析が重要だと考えます．なお，左右を判定できた試技は全体の14.3から25.5パーセントです．小さな移動や位置調整などもあり，方向の一致だけから，前衛がその割合で意識的に予測していたとまでは判断できません．');
}
// 10. Conclusion and next step
{
const s=base('まとめと今後の課題',10);
txt(s,'今回の成果',55,135,850,48,32,true,C.blue);
txt(s,'前衛の移動と打球方向の一致率を時点別に算出．\n全体と選手別の推移を示した．',55,202,850,108,31);
txt(s,'卒論で目指すこと',55,364,850,48,32,true,C.blue);
txt(s,'選手・対戦ペアに応じた\nポーチのタイミングの検討',55,431,850,97,35,true);
txt(s,'複数試合を収集し，動き始める時刻と\nポーチの成否・打球結果の関係を調べる．',55,575,850,80,28);
notes(s,'まとめと今後の課題',35,'まとめです．今回は，試合映像から前衛の移動方向を求め，打球方向との一致率を時点別，選手別に示しました．卒論では，選手や対戦ペアに応じたポーチのタイミングを検討したいと考えています．そのために複数試合を収集し，動き始める時刻に加えて，ポーチの成否や打球結果も記録します．一致率だけでなくプレーの結果との関係まで分析することを，次の課題とします．以上です．');
}
// 11. Coverage backup
{
const s=base('補足：方向を判定できた割合',11,true);
txt(s,'判定試技数 ÷ 各選手の全試技数',55,116,850,40,27);
const values=[['時点\n（秒）','P1\n全42試技','P2\n全56試技','全体\n全98試技'],...data.map(row=>[timeLabel(row[0].t),...row.map(q=>`${pct(q.coverage)}\n(${q.d} / ${q.n})`)])];
table(s,values,{y:165,rowH:88});
txt(s,'1.5％未満の移動は左右判定の対象外．追跡失敗とは別．',55,623,850,37,24);
notes(s,'補足：方向を判定できた割合',0,'一致率だけでは，どの程度の試技を評価できたかが分かりません．この表は，全試技のうち閾値を満たした割合です．1.5パーセント未満の移動を対象外としているため，ここに入らない試技すべてが追跡失敗という意味ではありません．');
}
// 12. Definitions backup
{
const s=base('補足：今回測っているもの',12,true);
txt(s,'位置',55,136,850,44,31,true,C.blue);
txt(s,'射影変換後の両足首の中点．横方向をコート幅で正規化．',55,192,850,77,28);
txt(s,'移動方向',55,308,850,44,31,true,C.blue);
txt(s,'各評価時点の直前約0.1秒間の横変位で判定．\n小さな揺れへの対策として3フレームの中央値を使用．',55,364,850,98,28);
txt(s,'評価対象',55,509,850,44,31,true,C.blue);
txt(s,'打球方向と前衛の移動方向が一致するか．\n初動時刻・意図・ポーチ成功率は今回の測定対象外．',55,565,850,94,28);
notes(s,'補足：今回測っているもの',0,'本研究の今回の値は，各時点の直前約0.1秒の変位です．動き出した最初の時刻を検出した結果ではありません．方向の一致と，選手が意識的に予測できていたか，あるいはポーチが成功したかは区別します．手前・奥の左右は本人視点に揃えています．','実装：scripts/86_build_court_plane_preimpact_features.py，scripts/87_summarize_court_plane_directional_motion.py．約59.94fps，100msに対応する窓は6フレーム．中央値フィルタの半径は1フレーム．');
}
// 13. Future evaluation backup
{
const s=base('補足：ポーチのタイミングの検証',13,true);
txt(s,'追加する記録',55,135,850,48,32,true,C.blue);
txt(s,'動き始める時刻，ポーチの実施と成否，打球結果\n前衛・打者の開始位置，選手・対戦ペア',55,199,850,106,29);
txt(s,'分析の単位',55,350,850,48,32,true,C.blue);
txt(s,'同じ選手・対戦ペアの複数試合で比較する．\n移動途中の反転や戻りも，映像で確認する．',55,414,850,106,29);
txt(s,'タイミングとプレーの結果がどう関係するかを評価',55,571,850,74,31,true);
notes(s,'補足：ポーチのタイミングの検証',0,'最大の一致率を示す時点を，最適なポーチ開始時刻とは呼びません．今後はポーチを実施した場面を定義し，動き始める時刻，成否，打球結果を追加します．フェイントなどの意図を映像から決めつけず，まず横移動，反転，戻りといった観察可能な動きを記録します．同じ選手や対戦ペアの複数試合を用いて，開始位置などの条件も整理した上で，タイミングと結果の関係を調べます．');
}
// 14. References
{
const s=base('参考文献',14,true);
txt(s,'［1］Jocher, G., Qiu, J.\n　　 Ultralytics YOLO11',55,160,850,115,30);
txt(s,'https://github.com/ultralytics/ultralytics',95,294,810,52,25,false,C.blue);
txt(s,'［2］Hartley, R., Zisserman, A.\n　　 Multiple View Geometry in Computer Vision\n　　 2nd ed., Cambridge University Press, 2004.',55,401,850,178,28);
notes(s,'参考文献',0,'姿勢推定にはUltralytics YOLO11を使用しました．射影変換の参考文献としてHartleyとZissermanの書籍を記載しています．','提出予稿の参考文献を踏襲．');
}

assert.equal(p.slides.items.length,14);
assert.equal(noteEntries.slice(0,10).reduce((n,x)=>n+x.sec,0),400);
const candidatePath=path.join(buildDir,`candidate_${revision}.pptx`);
await(await PresentationFile.exportPptx(p)).save(candidatePath);
const finalPath=path.join(outputDir,`23238007R_中間発表_画像差し替え版_${revision}.pptx`);
const result=await finalizePresentation({
 workspaceDir,candidatePath,finalPath,pythonExecutable:python,
 integrityValidatorPath:path.join(skillDir,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(skillDir,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','9144000,6858000','--validate-heading-fit',...tableOwners.flatMap(n=>['--require-native-table-slide',String(n)])],
 requiredNativeTableOwnerSlides:tableOwners,fontPolicy:{basis:'design',families:[font]},
 verifyArtifactToolImport:true,receiptPath:path.join(buildDir,`validation_${revision}.json`)
});
console.log(JSON.stringify(result));
const imported=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const previewDir=path.join(buildDir,`render_${revision}`);await fs.mkdir(previewDir,{recursive:true});
for(let i=0;i<imported.slides.items.length;i++){
 const png=await imported.export({slide:imported.slides.items[i],format:'png',scale:1.25});
 await fs.writeFile(path.join(previewDir,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));
 console.log('rendered '+(i+1));
}
await fs.writeFile(path.join(outputDir,'発表原稿.md'),'# 中間発表原稿\n\n本編10枚．時間配分の合計は6分40秒，残り20秒を切り替え等の余裕とする．実際の時間は読み合わせで調整する．\n\n'+noteEntries.map((n,i)=>`## ${i+1}．${n.title}${n.sec?'（'+n.sec+'秒）':'（質疑用）'}\n\n${n.script}\n`).join('\n'));
console.log(finalPath);
