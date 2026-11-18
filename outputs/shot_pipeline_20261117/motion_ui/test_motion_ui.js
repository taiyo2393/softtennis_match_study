"use strict";
// UIを開かず、保存される座標変換・計時・CSVの仕様を検証する。追加パッケージ不要。
const assert = require("node:assert/strict");
require("./app.js");
const C = globalThis.MotionCore;
const expectedN = ["straight_front","right_front","right_lateral","right_back","straight_back","left_back","left_lateral","left_front"];
const expectedF = ["straight_back","left_back","left_lateral","left_front","straight_front","right_front","right_lateral","right_back"];
C.DIRS.forEach((dir,i) => {
  assert.equal(C.convert(dir,"N"),expectedN[i]);
  assert.equal(C.convert(dir,"F"),expectedF[i]);
});
assert.equal(C.dir3("left_front"),"front");
assert.equal(C.dir3("right_back"),"back");
assert.equal(C.dir3("left_lateral"),"lateral");
assert.equal(C.convert("","N"),"");
let t = {ms:0,last:null};
C.touch(t,1000); assert.equal(C.elapsed(t,3000),2000);
C.touch(t,4000); assert.equal(t.ms,3000);
C.touch(t,64000); assert.equal(t.ms,3000); // ちょうど60秒の無操作全体を除外
C.touch(t,65000); assert.equal(t.ms,4000);
C.suspend(t,66000); assert.equal(t.ms,5000);
C.touch(t,900000); assert.equal(t.ms,5000); // 他の件の作業時間は加算しない
const l = {motion:"compound",screen_dir_final:"up_left",screen_dir_prelim:"up_right",feint:"yes",confidence:"high",note:'引用符"、カンマ,\n改行',saved_at:"2026-11-18T00:00:00Z"};
assert.equal(C.valid(l),"");
assert.match(C.valid({...l,screen_dir_prelim:""}),/予備/);
assert.match(C.valid({...l,motion:"unknown",unknown_reason:""}),/理由/);
assert.match(C.valid({...l,motion:"single"}),/複/);
const item = {id:"test",point:7,contact_frame:123,front_side:"F"};
const row = C.row(item,l,12500,"本人");
assert.equal(row.player_dir8_final,"right_back");
assert.equal(row.player_dir8_prelim,"left_back");
assert.equal(row.player_dir3_final,"back");
assert.equal(row.seconds,"12.500");
const none = C.row(item,{...l,motion:"none"},0,"");
assert.equal(none.screen_dir_final,""); assert.equal(none.player_dir8_prelim,"");
const text = C.csv([row]);
assert.ok(text.startsWith("\ufeff")); assert.ok(text.includes('引用符""、カンマ,\n改行'));
assert.equal(Object.keys(row).length,17);
console.log("方向変換16通り・3方向・計時の境界・必須項目・CSVの検証成功");
