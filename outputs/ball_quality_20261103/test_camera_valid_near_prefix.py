import unittest
import numpy as np
from camera_valid_near_prefix import CameraValidNearPrefix
from test_near_prefix_initialization import rows


class CameraEligibility(unittest.TestCase):
    def make(self):
        return CameraValidNearPrefix(np.diag([1/3840,1/2160,1.]),[(10,20),(30,40)])

    def step(self,tr,f):
        return tr.step(f,rows(f),[np.full((3,8),1/8,np.float32)]*2)

    def test_uneligible_frames_do_not_count_toward_confirmation(self):
        tr=self.make()
        for f in range(17):self.assertEqual(self.step(tr,f),[])
        self.assertEqual(len(self.step(tr,17)),2)

    def test_leaving_view_clears_state_and_needs_new_confirmation(self):
        tr=self.make()
        for f in range(21):self.step(tr,f)
        self.assertTrue(tr.states)
        self.assertEqual(self.step(tr,21),[])
        self.assertFalse(tr.states)
        for f in range(30,37):self.assertEqual(self.step(tr,f),[])
        self.assertEqual(len(self.step(tr,37)),2)


if __name__=='__main__':unittest.main()
