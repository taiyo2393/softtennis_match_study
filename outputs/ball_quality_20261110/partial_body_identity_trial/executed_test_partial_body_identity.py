import unittest
import numpy as np
from partial_body_identity import body_feature,BodyIdentity,distance


def person(x=50,index=0):
    k=np.zeros((17,3));k[:,2]=1
    k[5,:2]=[x-4,10];k[6,:2]=[x+4,10];k[11,:2]=[x-4,20];k[12,:2]=[x+4,20]
    return dict(box=[x-6,0,x+6,30],keypoints=k,pose_index=index,eligible_not_crop_truncated=True)


class EvidenceSafety(unittest.TestCase):
    def test_foreground_pixels_are_removed_instead_of_used_as_body_color(self):
        im=np.zeros((40,100,3),np.uint8);im[:]=(0,0,255)
        p=person();clean,_,_=body_feature(im,p)
        im[:,50:]=(255,255,255)
        mixed,_,_=body_feature(im,p)
        masked,count,original=body_feature(im,p,[[50,0,100,40]])
        self.assertGreater(distance(clean,mixed),0)
        self.assertAlmostEqual(distance(clean,masked),0,places=6)
        self.assertLess(count,original)

    def test_fully_masked_body_has_no_fallback(self):
        im=np.full((40,100,3),100,np.uint8)
        value,count,_=body_feature(im,person(),[[0,0,100,40]])
        self.assertIsNone(value);self.assertEqual(count,0)

    def test_unavailable_body_does_not_create_identity(self):
        a=np.zeros(192,np.float32);a[0]=1
        tr=BodyIdentity({'far_rear':[a]},[],{'far_rear':person()},10)
        self.assertEqual(tr.step(8,[person(48)],[None],set()),{})

    def test_known_near_pose_cannot_be_selected_as_far(self):
        a=np.zeros(192,np.float32);a[0]=1
        tr=BodyIdentity({'far_rear':[a]},[],{'far_rear':person()},10)
        self.assertEqual(tr.step(8,[person(48)],[a],{0}),{})

    def test_reverse_motion_and_one_to_one_assignment(self):
        a=np.zeros(192,np.float32);a[0]=1
        tr=BodyIdentity({'far_rear':[a]},[],{'far_rear':person()},10)
        out=tr.step(8,[person(48)],[a],set())
        self.assertEqual(set(out),{'far_rear'})
        self.assertGreater(tr.states['far_rear']['velocity'][0],0)
        self.assertEqual(set(tr.step(6,[person(46)],[a],set())),{'far_rear'})


if __name__=='__main__':unittest.main()
