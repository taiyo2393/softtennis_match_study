import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {Presentation,PresentationFile,FileBlob} from '@oai/artifact-tool';
const build=path.dirname(fileURLToPath(import.meta.url)),workspaceDir=path.dirname(build),study=path.resolve(workspaceDir,'../..');
const out=path.join(workspaceDir,'output'),assets=path.join(out,'実映像画像');
const skill='C:/Users/ta239/.codex/plugins/cache/openai-primary-runtime/presentations/26.904.11930/skills/presentations';
const python='C:/Users/ta239/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
process.env.RUNTIME_NODE_MODULES='C:/Users/ta239/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
process.env.RUNTIME_NODE=process.execPath;process.env.RUNTIME_PYTHON=python;
const {finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const rev=process.argv[2]??'v9';
const seminarUrl='https://web.plaud.ai/s/pub_7a5d8703-66bd-4619-abe1-9958d2bc9bf9::RNfpOMLQ2u3hYVHYsrDFe3a_HBR_lUuWXKagJL--Ns1qNWFpDlR11qbtLeVkapWJlhNRkn-gMf6a1R-y0Y4dIwI';
const seminarSource='研究ゼミ：朝隈（2026-09-16）、ユーザー提供のAI要約を2026-09-18にブラウザで確認。目的・価値の整理に使用。要約中の数値は現在のCSVと一致しないため転記しない。\n'+seminarUrl;
const source=path.join(out,'23238007R_中間発表_チェック04対応_v8.pptx');
const original=await PresentationFile.importPptx(await FileBlob.load(source));
await fs.writeFile(path.join(build,'check04_source.ndjson'),(await original.inspect({kind:'slide,layout,textbox,table,image,notes',maxChars:300000})).ndjson);
// Full re-layout is intentional: the new rubric requires actual 28pt body/table
// text (37.34 CSS px), named footers and a four-month plan. Preserve 4:3,
// Meiryo, navy/white styling, source photographs and the verified observations.
const eventPath=path.join(study,'analysis/match1_court_plane_directional_motion_t0p015_event_timepoints.csv');
const csv=(await fs.readFile(eventPath,'utf8')).trim().split(/\r?\n/),head=csv.shift().split(',');
const rows=csv.map(l=>Object.fromEntries(l.split(',').map((v,i)=>[head[i],v])));
assert.equal(rows.length,392);assert.equal(new Set(rows.map(r=>r.event_id)).size,98);
const times=[-.5,-.3,-.2,-.1];
const data=times.map(t=>['P1','P2','ALL'].map(id=>{
 const rr=rows.filter(r=>Number(r.timepoint_sec)===t&&(id==='ALL'||r.target_front_player_id===id));
 const n=rr.length,d=rr.filter(r=>r.movement_is_directional==='1').length,k=rr.filter(r=>r.direction_matches_actual==='1').length;
 return {id,t,n,d,k,rate:k/d,coverage:d/n};
}));
assert.deepEqual(data.map(r=>[r[2].d,r[2].k]),[[14,6],[25,16],[15,11],[22,15]]);
const p=Presentation.create({slideSize:{width:960,height:720}}),font='Meiryo',blue='#214F76',ink='#182126';
const tableOwners=[],notes=[],audit=[];
function text(s,value,x,y,w,h,size=38,color=ink,align='left',fill='none'){
 const q=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill,line:{fill:'none',width:0}});
 q.text=value;q.text.style={typeface:font,fontSize:size,bold:true,color,alignment:align,verticalAlignment:'middle',autoFit:'none'};return q;
}
function base(title){const s=p.slides.add();s.background.fill='#FFFFFF';
 text(s,title,40,24,880,74,44,blue);
 text(s,'23238007　朝隈 大耀',40,678,740,27,18,ink);
 text(s,String(p.slides.items.length),840,667,80,40,28,ink,'right');return s;}
function note(s,sec,script,sourceExtra=''){
 if(p.slides.items.length>=4&&p.slides.items.length<=11&&sec)sec-=5;
 const title=s===p.slides.items[0]?'題目':s.shapes.items[0].text.toString();
 notes.push({slide:p.slides.items.length,title,sec,script});
 s.speakerNotes.textFrame.setText(`${sec?'発表目安 '+sec+'秒':'質疑用'}\n\n${script}\n\n出典・補足（読み上げない）\n${eventPath}\n${sourceExtra}`);
}
function table(s,vals,widths,{y=170,rowH=86,headerH=rowH,highlights=[]}={}){
 const t=s.tables.add({rows:vals.length,columns:vals[0].length,left:40,top:y,width:880,height:headerH+(vals.length-1)*rowH,columnWidths:widths,values:vals});
 t.styleOptions={headerRow:false,bandedRows:false};t.borders.assign({style:'solid',fill:'#8093A1',width:1});
 for(let r=0;r<vals.length;r++){t.rows[r].height=r===0?headerH:rowH;for(let c=0;c<vals[0].length;c++){
 const a=t.getCell(r,c);a.fill=r===0?'#EAF0F4':'#FFFFFF';
 a.text.style={typeface:font,fontSize:38,bold:true,color:ink,alignment:'center',verticalAlignment:'middle',autoFit:'none'};
 }}
 for(const [r,c] of highlights){t.getCell(r,c).fill='#EAF0F4';t.getCell(r,c).text.style={typeface:font,fontSize:38,bold:true,color:blue,alignment:'center',verticalAlignment:'middle',autoFit:'none'};}
 tableOwners.push(p.slides.items.length);return t;
}
async function photo(s,file,x,y,w,h,alt){s.images.add({blob:await fs.readFile(path.join(assets,file)),contentType:'image/png',alt,fit:'contain',position:{left:x,top:y,width:w,height:h}});}
const pct=x=>(100*x).toFixed(1)+'％',tm=x=>'−'+Math.abs(x).toFixed(1);
// 1. Main topic is stated in the opening, well before the 15% deadline.
{
 const s=p.slides.add();s.background.fill='#FFFFFF';
 text(s,'学部・中間発表',50,65,860,60,38,blue,'center');
 text(s,'ソフトテニスの試合映像を用いた\n動作予測手法の提案',40,190,880,150,46,blue,'center');
 text(s,'23238007　朝隈 大耀',40,400,880,60,38,ink,'center');
 text(s,'佐賀大学 理工学部',40,485,880,58,38,ink,'center');
 text(s,'指導教員　中山 功一 教授',40,573,880,58,38,ink,'center');
 note(s,20,'ソフトテニスの試合映像を用いた動作予測手法の提案について，朝隈が発表します．目指すのは，前衛が相手に動きを読まれにくく，球にも届くタイミングを検討できるようにすることです．その第一段階として，打球前の動きと打球方向の対応を分析しました．',seminarSource);
}
// 2. The photographed roles are labelled directly; no pointer is necessary.
{
 const s=base('目的：前衛が出るタイミングの検討');
 text(s,'相手に読まれにくく，球にも届く出方へ',40,113,880,53,38,blue);
 await photo(s,'画像①_コート全景_F145668.png',40,181,650,366,'原動画F145668。手前前衛は橙色、手前後衛は黒色。');
 text(s,'前衛',265,314,110,51,38,blue,'center','#FFFFFF');
 text(s,'後衛',401,489,110,51,38,blue,'center','#FFFFFF');
 text(s,'早い出方\n読まれる？\n\n遅い出方\n届かない？',706,185,214,340,38);
 text(s,'ポーチ：横へ出て相手の球を捉えるプレー',40,574,880,56,38,blue);
 note(s,40,'ダブルスでは，ネット付近の前衛と，後方で打ち合う後衛がペアを組みます．前衛が横へ出て，相手の球を途中で捉えるプレーがポーチです．出るのが早いと動きを読まれて逆を突かれる可能性があり，遅いと球に届かない可能性があります．そこで，この相手にはいつ出ればよいかを，選手や指導者が映像と数値で検討できるようにしたいと考えました．このタイミングの判断を支援することが研究の目的です．',seminarSource+'\n写真：data/raw/match1.mp4，F145668。写真自体は早い・遅い出方の成功失敗を示す実験例ではなく、役割の説明。右欄は検証する競技上の課題。');
}
{
 const s=base('今回の分析と目指す活用');
 table(s,[['段階','調べる内容・使い道'],['今回','打球前の動きと方向を比較\n時点別・選手別の一致率'],['卒論','動き出す時刻と成否を比較\n相手・開始位置の違いも考慮'],['活用','前衛：相手に応じた出方を検討\n指導者：映像と時刻で助言']],[230,650],{y:145,rowH:116});
 text(s,'今回の一致率だけで最適時刻は決めない',40,614,880,50,38,blue);
 note(s,35,'そのために，まず打球前のどの区間で動きと打球方向が対応するかを調べました．選手ごとの違いを比較するための第一段階です．ただし，一致率が高い区間が，そのままポーチに最適なタイミングではありません．卒論では，動き出した時刻とポーチの成功・失敗を，相手や開始位置と併せて分析します．前衛が相手に応じた出方を練習し，指導者が具体的な時刻と映像で助言できることを目指します．',seminarSource);
}
{
 const s=base('今期の実施内容：98試技の分析');
 text(s,'2025年 全日本選手権 男子決勝・1試合',40,115,880,54);
 table(s,[['対象前衛','確認済み試技'],['P1','42'],['P2','56'],['全体','98']],[440,440],{y:190,rowH:92});
 text(s,'1試技＝相手後衛の1回の打球',40,579,880,62);
 note(s,30,'今期は，2025年全日本ソフトテニス選手権大会の男子決勝から，98試技を整理しました．1試技は相手後衛の1回の打球です．前衛は2人で，P1が42試技，P2が56試技です．チェンジコート後も同じ選手番号を保ち，手前と奥をまとめて分析しました．','今期とは今回の中間発表対象期間。開始月を推測して付記していない。P1/P2は内部識別番号。');
}
{
 const s=base('実施①：前衛の足首位置の抽出');
 text(s,'姿勢推定：画像から身体の点を求める',40,115,880,54);
 await photo(s,'画像②_足首追跡_4K原画から_F399.png',75,181,810,456,'原動画F399と保存済みYOLO11m-poseの足首位置。黄枠は前衛、紫丸は左右の足首。');
 text(s,'黄枠：対象前衛　紫丸：足首',92,190,755,54,38,ink,'center','#FFFFFF');
 note(s,30,'前衛の位置を読むため，姿勢推定モデルのYOLO11m-poseを使用しました．姿勢推定とは，画像から足首などの身体の点を求める処理です．黄枠が対象前衛で，紫の丸が両足首です．別の選手に追跡が移った場面などは，人手で確認して修正しました．','YOLO11m-pose［1］。画像の人や足首を生成AIで描き直していない。出典：実映像画像/画像出典_高解像度版.json。');
}
{
 const s=base('実施②：映像と打球情報の対応付け');
 table(s,[['情報','今回の処理'],['打球時刻','接触した画像を\n人手で確定'],['打球方向','対象前衛から見た\n左右を確認'],['足首位置','コートの座標へ\n変換し，中点を計算']],[240,640],{y:141,rowH:116});
 text(s,'射影変換：映像の遠近の見え方を補正',40,611,880,49);
 note(s,35,'ボールがラケットに当たる瞬間をインパクトと呼び，その画像を人手で確定しました．打球方向も対象前衛本人から見た左右で記録しています．足首位置には，遠近の見え方を補正する射影変換を適用しました．左右それぞれの足首をコートの座標に変換し，その中点を計算します．今回の結果は，この足首中点に基づきます．','射影変換［2］。両足首を個別に変換後、中点を計算。');
}
{
 const s=base('実施③：足首中点の変位方向の判定');
 text(s,'変位＝位置の変化．打球時刻を0秒とする',40,111,880,55);
 table(s,[['評価時点','比較する区間'],['−0.5秒','−0.6秒 ～ −0.5秒'],['−0.3秒','−0.4秒 ～ −0.3秒'],['−0.2秒','−0.3秒 ～ −0.2秒'],['−0.1秒','−0.2秒 ～ −0.1秒']],[275,605],{y:180,rowH:78});
 text(s,'横変位がコート幅の1.5％以上：左右を判定\n1.5％未満：左右判定の対象外',40,577,880,90);
 note(s,35,'各評価時点の直前0.1秒について，足首中点の横方向の位置の変化，つまり変位を求めます．例えば0.3秒前の判定には，0.4秒前と0.3秒前を比較します．横変位がコート幅の1.5パーセント以上の場合だけ，左か右を判定します．それ未満は対象外です．これは動き始めた時刻の検出ではなく，指定した時点での方向の判定です．','約59.94fpsで6フレーム間隔。3フレーム中央値を使用。手前/奥は本人視点の左右に統一。');
}
{
 const s=base('結果①：全体の方向の一致率');
 text(s,'一致率＝一致数 ÷ 左右の判定試技数',40,113,880,55);
 table(s,[['時点（秒）','判定数','一致数','一致率'],...data.map(r=>{const q=r[2];return[tm(q.t),String(q.d),String(q.k),pct(q.rate)]})],[220,200,200,260],{y:185,rowH:81,highlights:[[3,3],[4,3]]});
 text(s,'−0.2秒までは上昇，−0.1秒では低下',40,611,880,53,38,blue);
 note(s,40,'全体の一致率です．足首中点の変位方向と打球方向が一致した数を，左右を判定できた数で割っています．0.5秒前は42.9パーセント，0.3秒前は64.0パーセント，0.2秒前は73.3パーセントでした．ただし0.1秒前では68.2パーセントに下がります．打球に近いほど常に上がるわけではありません．この値は，前衛本人の予測の正答率と同じではありません．');
}
{
 const s=base('結果②：選手別の方向の一致率');
 text(s,'一致率（一致数／判定数）',40,113,880,55);
 table(s,[['時点（秒）','P1：全42試技','P2：全56試技'],...data.map(r=>[tm(r[0].t),...r.slice(0,2).map(q=>`${pct(q.rate)} (${q.k}/${q.d})`)])],[240,320,320],{y:180,rowH:86,highlights:[[4,1],[4,2]]});
 text(s,'−0.1秒の推移に違い．能力差とは未確定',40,619,880,49,38,blue);
 note(s,40,'選手別では，今回の4時点についてP1は50.0パーセントから81.8パーセントへ上昇しました．一方，P2は0.2秒前が75.0パーセントで，0.1秒前には54.5パーセントへ下がっています．全体集計だけでは見えにくい違いを示せました．ただし各時点で対象になる試技は同じではなく，数も少ないため，選手の能力差が分かったとは判断していません．');
}
{
 const s=base('結果③：左右を判定できた割合');
 text(s,'判定率＝左右の判定数 ÷ 全98試技',40,113,880,55);
 table(s,[['時点（秒）','判定数／全試技','判定率'],...data.map(r=>{const q=r[2];return[tm(q.t),`${q.d}／${q.n}`,pct(q.coverage)]})],[260,330,290],{y:185,rowH:81});
 text(s,'小さな変位は対象外．全てが追跡失敗ではない',40,611,880,53,38,blue);
 note(s,30,'一致率と併せて，どの程度の試技を判定できたかも確認しました．判定率は14.3から25.5パーセントでした．この値が小さい主な理由は，コート幅の1.5パーセント以上という条件を設けたためです．対象外の全てが追跡失敗ではありません．一致率だけを示さず，判定できた範囲も併せて報告する必要があります．');
}
{
 const s=base('考察：今回の結果と測定上の課題');
 table(s,[['結果の見方','解釈で注意すること'],['一致率の時点差','対象試技も入れ替わる'],['方向が不一致','フェイント・戻りも含む'],['足首中点の変位','足の開閉も含まれる']],[440,440],{y:156,rowH:108});
 text(s,'一致率だけで予測の成否は断定しない',40,612,880,55,38,blue);
 note(s,40,'時点による一致率の違いはありますが，集計対象も入れ替わります．例えば0.2秒前の15試技と0.1秒前の22試技で，共通するのは11試技です．不一致にはフェイントや戻りも含まれ得るため，予測の失敗とは限りません．また，足を閉じるだけでも中点は動きます．身体移動との区別を検証し，開始位置や動きの反転も整理した上で，タイミングと成否の関係を調べます．',seminarSource+'\n既存集計は再解析していない。同一event_id・timepoint_secの重複0件。判定対象41試技、延べ76件。各時点で同じ母集団を追跡した一致率ではない。フェイントなどの意図は映像のみから確定しない。');
}
{
 const s=base('今後4か月の計画（案）');
 table(s,[['時期','実施すること'],['1か月目','足の開閉と身体移動を区別\n人手判定との一致を検証'],['2か月目','3試合以上を収集\n開始位置・打球結果を記録'],['3か月目','選手別・対戦相手別に比較\n動き出す時刻と成否を集計'],['4か月目','別試合でも傾向を確認\n卒論の執筆・修正']],[230,650],{y:128,rowH:116,headerH:64});
 note(s,40,'発表後4か月の計画案です．最初に，足の開閉と身体移動を区別できるか，人手判定と比較します．次に，利用条件を確認した上で3試合以上へ広げ，開始位置や打球結果を記録します．3か月目には，選手別や対戦相手別に，動き出す時刻とポーチの成否を分析します．最後に別試合でも確認し，卒論にまとめます．ポーチは，前衛が横へ出て相手の球を途中で捉えるプレーです．','計画案。未実施。ポーチ実施・成功の定義を収集前に確定。未達時は不足理由と観測範囲を報告する。');
}
{
 const s=base('研究の特色と卒論の到達目標');
 table(s,[['観点','目指す成果'],['新規性の焦点','相手との組合せ別に\n動く時刻と成否を対応付け'],['有効性','前衛・指導者が\n相手に応じた出方を検討'],['到達基準\n（案）','3試合以上で検証\n人手との誤差・判定率を提示'],['最終成果','タイミング別の成功・失敗を\n映像と数値で説明']],[275,605],{y:128,rowH:116,headerH:64});
 note(s,30,'目指す成果は，この相手にはどのタイミングで出ると成功しやすいか，失敗しやすいかを，映像と数値で説明できることです．前衛の練習や，指導者の具体的な助言に役立てたいと考えています．今回はその第一段階として，打球前の方向の一致を示しました．卒論では3試合以上に拡大し，測定の誤差と判定率も含めて，タイミングと成否の関係を検証します．以上です．',seminarSource+'\n新規性は検証する焦点。先行研究との未重複、利用効果、最適タイミングの確定は未実証。目標は提案段階であり、観察研究から因果的な最適時刻を断定しない。');
}
{
 const s=base('補足：選手別の判定率');
 text(s,'判定率（判定数／各選手の全試技）',40,113,880,55);
 table(s,[['時点','P1：全42試技','P2：全56試技'],...data.map(r=>[tm(r[0].t)+'秒',...r.slice(0,2).map(q=>`${pct(q.coverage)} (${q.d}/${q.n})`)])],[180,350,350],{y:180,rowH:86});
 text(s,'手前・奥を，同じ選手番号で統合',40,619,880,49);
 note(s,0,'選手別に，閾値を満たした試技の割合を示しています．一致率とは分母が違います．');
}
{
 const s=base('参考文献');
 text(s,'［1］Jocher, G., Qiu, J.\nUltralytics YOLO11\nhttps://github.com/ultralytics/ultralytics',50,140,850,145,28);
 text(s,'［2］Hartley, R., Zisserman, A.\nMultiple View Geometry in Computer Vision\n2nd ed., Cambridge University Press, 2004.',50,338,850,145,28);
 text(s,'映像：2025年 全日本ソフトテニス選手権大会\n男子決勝（研究用保存映像 match1.mp4）',50,540,850,86,26);
 note(s,0,'使用した姿勢推定モデルと射影変換の参考文献です．','既存予稿・既存スライドの文献を保持。ここは後で参照するための文献ページで、28pt未満を許容するS3例外。');
}
assert.equal(p.slides.items.length,15);
const spokenChars=notes.slice(0,13).reduce((n,r)=>n+[...r.script.replace(/\s/g,'')].length,0);
const sec=notes.slice(0,13).reduce((n,r)=>n+r.sec,0);assert.equal(sec,405);assert.ok(spokenChars<=2100);
await fs.writeFile(path.join(build,`check04_metrics_${rev}.json`),JSON.stringify({source,seminarUrl,mainSlides:13,totalSlides:15,spokenChars,estimatedAtFiveCharsPerSecond:spokenChars/5,allocatedSeconds:sec,bodyFontPt:38*.75,pageFontPt:28*.75,footerFontPt:18*.75,completedWorkMainSlides:[4,5,6,7,8,9,10,11],completedWorkAllSlides:[4,5,6,7,8,9,10,11,14],data,tableOwners},null,2));
await fs.writeFile(path.join(out,`発表原稿_研究目的明確化_${rev}.md`),'# 発表原稿\n\n本編13枚、配分6分45秒。最後の2枚は質疑用。実際の所要時間は読み合わせで確認してください。\n\n'+notes.map(n=>`## ${n.slide}．${n.title}（${n.sec?n.sec+'秒':'質疑用'}）\n\n${n.script}\n`).join('\n'));
const candidatePath=path.join(build,`candidate_check04_${rev}.pptx`);await(await PresentationFile.exportPptx(p)).save(candidatePath);
const finalPath=path.join(out,`23238007R_中間発表_研究目的明確化_${rev}.pptx`);
await finalizePresentation({workspaceDir,candidatePath,finalPath,pythonExecutable:python,integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','9144000,6858000','--validate-heading-fit',...tableOwners.flatMap(n=>['--require-native-table-slide',String(n)])],requiredNativeTableOwnerSlides:tableOwners,fontPolicy:{basis:'design',families:[font]},verifyArtifactToolImport:true,receiptPath:path.join(build,`validation_check04_${rev}.json`)});
const final=await PresentationFile.importPptx(await FileBlob.load(finalPath));
const render=path.join(build,`render_check04_${rev}`);await fs.mkdir(render,{recursive:true});
for(let i=0;i<15;i++){const png=await final.export({slide:final.slides.items[i],format:'png',scale:1.25});await fs.writeFile(path.join(render,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await png.arrayBuffer()));console.log('rendered '+(i+1));}
console.log(JSON.stringify({finalPath,spokenChars,sec}));
