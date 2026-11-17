from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree as E
import posixpath
base=Path(__file__).parent
source=Path('C:/Users/ta239/Desktop/中山研/23238007R_MID.pptx')
ns={'p':'http://schemas.openxmlformats.org/presentationml/2006/main','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
with ZipFile(source) as z: data={n:z.read(n) for n in z.namelist()}
def xml(n): return E.fromstring(data[n])
def put(n,v): data[n]=E.tostring(v,xml_declaration=True,encoding='UTF-8',standalone=True)
pres=xml('ppt/presentation.xml'); ids=pres.find('p:sldIdLst',ns); keep=list(ids)[2:5]; keepids={v.get('{'+ns['r']+'}id') for v in keep}
for e in list(ids):
 if e not in keep: ids.remove(e)
rels=xml('ppt/_rels/presentation.xml.rels'); removed=set(); selected=[]
for r in list(rels):
 if r.get('Type').endswith('/slide'):
  target=posixpath.normpath(posixpath.join('ppt',r.get('Target')))
  if r.get('Id') in keepids: selected.append(target)
  else:
   rels.remove(r); removed.add(target)
   rp=posixpath.join(posixpath.dirname(target),'_rels',posixpath.basename(target)+'.rels')
   if rp in data:
    for sr in xml(rp):
     if sr.get('Type').endswith('/notesSlide'):
      nt=posixpath.normpath(posixpath.join(posixpath.dirname(target),sr.get('Target')))
      removed.add(nt); removed.add(posixpath.join(posixpath.dirname(nt),'_rels',posixpath.basename(nt)+'.rels'))
    removed.add(rp)
for e in list(pres):
 if E.QName(e).localname in ('custShowLst','extLst'): pres.remove(e)
put('ppt/presentation.xml',pres); put('ppt/_rels/presentation.xml.rels',rels)
for name in selected:
 s=xml(name); children=s.find('p:timing/p:tnLst/p:par/p:cTn/p:childTnLst',ns)
 seq=children.find('p:seq',ns); assert seq is not None
 before=E.tostring(seq,method='c14n')
 for e in list(children):
  if e is not seq: children.remove(e)
 assert before==E.tostring(seq,method='c14n')
 assert len(s.xpath('.//p:animMotion',namespaces=ns))==2
 assert len(s.xpath('.//p:cTn[@nodeType="clickEffect"]',namespaces=ns))==1
 trans=s.find('p:transition',ns)
 if trans is not None:
  trans.attrib.pop('advTm',None); trans.set('advClick','1')
 put(name,s)
 print(name,[(e.get('path'), e.find('p:cBhvr/p:tgtEl/p:spTgt',ns).get('spid')) for e in s.xpath('.//p:animMotion',namespaces=ns)])
for n in removed: data.pop(n,None)
ct=xml('[Content_Types].xml')
for e in list(ct):
 if e.get('PartName','').lstrip('/') in removed: ct.remove(e)
put('[Content_Types].xml',ct)
app=xml('docProps/app.xml')
for e in app.iter():
 if E.QName(e).localname=='Slides': e.text='3'
put('docProps/app.xml',app)
with ZipFile(base/'three_candidate.pptx','w',ZIP_DEFLATED) as z:
 for n,b in data.items(): z.writestr(n,b)
print('3 slides; 1 click sequence and 2 original paths per slide; source unchanged.')
