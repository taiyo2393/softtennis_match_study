import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {spawnSync} from 'node:child_process';
import assert from 'node:assert/strict';
const build=path.dirname(fileURLToPath(import.meta.url));
const study=path.resolve(build,'../../..');
const dir=path.resolve(build,'../output/実映像画像');
await fs.mkdir(dir,{recursive:true});
const raw=path.join(study,'data/raw/match1.mp4');
const ffmpeg='C:/ffmpeg/bin/ffmpeg.exe';
const ffprobe='C:/ffmpeg/bin/ffprobe.exe';
const probe=spawnSync(ffprobe,['-v','error','-select_streams','v:0','-show_entries','stream=width,height,r_frame_rate,time_base,start_time','-of','json',raw],{encoding:'utf8'});
assert.equal(probe.status,0);
const stream=JSON.parse(probe.stdout).streams[0];
assert.equal(stream.r_frame_rate,'60000/1001');assert.equal(stream.time_base,'1/60000');
const entries=[];
function run(args){const r=spawnSync(ffmpeg,['-hide_banner','-loglevel','info',...args],{encoding:'utf8',maxBuffer:10e6});assert.equal(r.status,0,r.stderr);return r.stderr;}
async function rawFrame(name,frame,crop,event){
  const file=path.join(dir,name+'.png');
  const log=run(['-copyts','-ss',String(Math.max(0,(frame-60)*1001/60000)),'-i',raw,'-vf',`select=eq(pts\\,${frame*1001}),showinfo${crop?',crop='+crop.join(':'):''}`,'-frames:v','1','-update','1',file]);
  await fs.writeFile(path.join(build,name+'.extract.log'),log);
  assert.match(log,new RegExp('pts:\\s*'+frame*1001+'\\s'));
  assert.ok((await fs.stat(file)).size>10000);
  entries.push({file,source:raw,source_frame_zero_based:frame,source_pts:frame*1001,source_time_seconds:frame*1001/60000,event_id:event,crop_width_height_x_y:crop??null,processing:'Exact source PTS selection and crop only; no generated pixels or retouching'});
}
await rawFrame('画像①_コート全景_F145668',145668,null,'match1_direction_match1_remaining_01_182');
await rawFrame('画像③-A_インパクト前0.3秒_F104663',104663,[800,450,780,1060],'match1_direction_pilot_01_020');
await rawFrame('画像③-B_インパクト前0.2秒_F104669',104669,[800,450,780,1060],'match1_direction_pilot_01_020');
const preview=path.join(study,'data/pose/target_front/previews/match1_direction_match1_remaining_01_003.mp4');
const posefile=path.join(dir,'画像②_足首追跡_F399.png');
const poseLog=run(['-i',preview,'-vf','select=eq(n\\,24),crop=320:180:365:325','-frames:v','1','-update','1',posefile]);
await fs.writeFile(path.join(build,'pose399.extract.log'),poseLog);
entries.push({file:posefile,source:preview,local_frame_zero_based:24,source_frame_zero_based:399,event_id:'match1_direction_match1_remaining_01_003',crop_width_height_x_y:[320,180,365,325],processing:'Extracted from existing audited pose-preview video; original overlays retained',overlay:'Yellow: target box; magenta: ankle estimates; green: screen-space ankle midpoint guide (not the homography-transformed analysis coordinate)'});
await fs.writeFile(path.join(dir,'画像出典.json'),JSON.stringify({fps:'60000/1001',frame_index:'zero-based',entries,motion_example:{event_id:'match1_direction_pilot_01_020',impact_source_frame:104681,A_relative_seconds:(104663-104681)*1001/60000,B_relative_seconds:(104669-104681)*1001/60000,interval_seconds:6*1001/60000,median_displacement_court_width:.038573,threshold_court_width:.015,movement_direction_target_front_view:'right',note:'A/B are raw frames. The reported displacement uses the existing 3-frame median-filtered analysis, not a measurement from the cropped screenshots.'}},null,2));
console.log(JSON.stringify(entries,null,2));
