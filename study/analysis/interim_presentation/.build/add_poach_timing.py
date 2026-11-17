import sys, zipfile, re
import xml.etree.ElementTree as E

P='http://schemas.openxmlformats.org/presentationml/2006/main'
ns={'p':P}
def motion(node_id, shape_id, dx, dy, start, dur):
    return f'''<p:animMotion origin="layout" path="M 0 0 L {dx:.8f} {dy:.8f} E" pathEditMode="relative" rAng="0"><p:cBhvr additive="base"><p:cTn id="{node_id}" dur="{dur}" fill="hold"><p:stCondLst><p:cond delay="{start}"/></p:stCondLst></p:cTn><p:tgtEl><p:spTgt spid="{shape_id}"/></p:tgtEl><p:attrNameLst><p:attrName>ppt_x</p:attrName><p:attrName>ppt_y</p:attrName></p:attrNameLst></p:cBhvr></p:animMotion>'''
with zipfile.ZipFile(sys.argv[1]) as src, zipfile.ZipFile(sys.argv[2],'w',zipfile.ZIP_DEFLATED) as dst:
    for entry in src.infolist():
        data=src.read(entry.filename)
        if entry.filename in ['ppt/slides/slide1.xml','ppt/slides/slide2.xml','ppt/slides/slide3.xml']:
            xml=data.decode('utf-8'); root=E.fromstring(xml)
            ids={e.get('name'):e.get('id') for e in root.findall('.//p:cNvPr',ns)}
            early=entry.filename.endswith('slide1.xml')
            effects=motion(3,ids['target_front'],(135 if early else 80)/960,0,800 if early else 2900,1200 if early else 1200)
            effects+=motion(4,ids['ball'],(-45 if early else 220)/960,340/720,2600 if early else 1500,2000 if early else 2200)
            if entry.filename.endswith('slide3.xml'):
                effects=motion(3,ids['target_front'],104/960,0,1400,1420)
                effects+=motion(4,ids['ball'],132/960,204/720,1500,1320)
            timing=f'''<p:timing><p:tnLst><p:par><p:cTn id="1" dur="indefinite" restart="never" nodeType="tmRoot"><p:childTnLst><p:par><p:cTn id="2" dur="5000" fill="hold"><p:stCondLst><p:cond delay="0"/></p:stCondLst><p:childTnLst>{effects}</p:childTnLst></p:cTn></p:par></p:childTnLst></p:cTn></p:par></p:tnLst></p:timing>'''
            assert '<p:timing' not in xml
            if '<p:extLst' in xml:
                # A slide-level timing element belongs before the slide-level extLst.
                ext=root.find('p:extLst',ns)
                if ext is not None:
                    pos=xml.rfind('<p:extLst');xml=xml[:pos]+timing+xml[pos:]
                else:xml=xml.replace('</p:sld>',timing+'</p:sld>')
            else:xml=xml.replace('</p:sld>',timing+'</p:sld>')
            check=E.fromstring(xml)
            assert len(check.findall('.//p:animMotion',ns))==2
            data=xml.encode('utf-8')
        dst.writestr(entry,data)
