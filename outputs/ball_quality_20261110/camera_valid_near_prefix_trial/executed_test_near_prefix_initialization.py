"""Causal/ID invariants for the narrow prefix policy, using synthetic boxes."""
import unittest
import numpy as np
from near_prefix_initialization import NearPrefix


def rows(frame, reverse=False, changed_id=False):
    result=[]
    for tid,x,y in [(1,1500,1500),(2,2500,1800)]:
        result.append(dict(source_frame=str(frame),track_id=str(9 if changed_id and tid==1 else tid),
            x1=str(x-100),y1=str(y-500),x2=str(x+100),y2=str(y),
            bottom_center_x=str(x),bottom_center_y=str(y),confidence='.9'))
    return list(reversed(result)) if reverse else result


class PrefixInvariants(unittest.TestCase):
    def make(self):
        return NearPrefix(np.diag([1/3840,1/2160,1.]),True)

    def step(self,tr,f,**kw):
        rs=rows(f,**kw)
        # Same uniform, finite feature distances; exact row objects must survive.
        result=tr.step(f,rs,[np.full((3,8),1/8,dtype=np.float32) for r in rs])
        self.assertTrue(all(any(r is original for original in rs) for _,r,_,_ in result))
        return result

    def test_confirmation_does_not_backfill(self):
        tr=self.make()
        for f in range(7):self.assertEqual(self.step(tr,f),[])
        self.assertEqual(len(self.step(tr,7)),2)
        self.assertEqual(tr.events[0]['window_start'],0)
        self.assertEqual(tr.events[0]['frame'],7)

    def test_nonconsecutive_confirmation_resets(self):
        tr=self.make()
        for f in range(6):self.step(tr,f)
        for f in range(7,14):self.assertEqual(self.step(tr,f),[])
        self.assertEqual(len(self.step(tr,14)),2)

    def test_new_id_is_not_reacquired(self):
        tr=self.make()
        for f in range(8):self.step(tr,f)
        result=self.step(tr,8,changed_id=True)
        self.assertEqual([slot for slot,r,_,_ in result],['near_rear'])

    def test_long_gap_is_not_recovered(self):
        tr=self.make()
        for f in range(8):self.step(tr,f)
        self.assertEqual(tr.step(8,[],[]),[])
        self.assertEqual(tr.step(9,[],[]),[])
        self.assertEqual(self.step(tr,10),[])

    def test_candidate_order_does_not_change_seeds(self):
        a,b=self.make(),self.make()
        for f in range(8):
            ra=self.step(a,f)
            rb=self.step(b,f,reverse=True)
        self.assertEqual([(s,r['track_id']) for s,r,_,_ in ra],[(s,r['track_id']) for s,r,_,_ in rb])


if __name__=='__main__':unittest.main()
