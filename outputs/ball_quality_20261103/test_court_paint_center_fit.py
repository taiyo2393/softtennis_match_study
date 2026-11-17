import unittest
import cv2,numpy as np
from fit_match2_court_geometry_centers import fit_line_centers


class PaintCenters(unittest.TestCase):
    def test_straight_stripe_thickness_is_not_alignment_error(self):
        for width in [2,6,12]:
            image=np.zeros((120,400,3),np.uint8)
            cv2.line(image,(10,60),(390,60),(255,255,255),width)
            line,d=fit_line_centers(image,[[10,60],[390,60]])
            self.assertTrue(d['accepted'])
            self.assertLess(d['median_center_residual_px'],.1)
            self.assertLess(abs(line@[200,60,1]),.1)
    def test_absence_does_not_create_a_line(self):
        value,d=fit_line_centers(np.zeros((120,400,3),np.uint8),[[10,60],[390,60]])
        self.assertIsNone(value);self.assertFalse(d['accepted'])
    def test_short_white_object_does_not_supply_full_line_coverage(self):
        image=np.zeros((120,400,3),np.uint8)
        cv2.rectangle(image,(180,52),(220,68),(255,255,255),-1)
        line,d=fit_line_centers(image,[[10,60],[390,60]])
        self.assertFalse(d['accepted'])


if __name__=='__main__':unittest.main()
