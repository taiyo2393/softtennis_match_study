import unittest
import numpy as np
from match_frame_coordinate_adapter import FrameCoordinates


class Coordinates(unittest.TestCase):
    def test_native_box_roundtrip_keeps_ID_and_clock(self):
        adapter=FrameCoordinates(1920,1080)
        row=dict(source_frame=10546,timestamp_sec='175.942',track_id='59',x1=312.25,y1=520.5,x2=370.75,y2=660.25,bottom_center_x=341.5,bottom_center_y=660.25)
        converted=adapter.row(row)
        self.assertEqual(converted['y2'],1320.5)
        self.assertEqual(adapter.row(converted,to_canonical=False),row)
        self.assertEqual(converted['source_frame'],row['source_frame'])
        self.assertEqual(converted['track_id'],row['track_id'])
    def test_unknown_coordinates_stay_unknown(self):
        r=dict(bottom_center_x='',bottom_center_y=None,identity_status='body_only')
        self.assertEqual(FrameCoordinates(1920,1080).row(r),r)
    def test_keypoint_confidence_is_not_scaled(self):
        a=FrameCoordinates(1920,1080);k=np.array([[100.,200.,.7]],np.float32)
        np.testing.assert_array_equal(a.keypoints(k),[[200.,400.,np.float32(.7)]])
        np.testing.assert_array_equal(a.keypoints(a.keypoints(k),False),k)
    def test_homography_rescaling_preserves_existing_calibration_projection(self):
        a=FrameCoordinates(1920,1080);H=np.array([[.01,0,1],[0,.02,2],[.0001,.0002,1.]])
        native=np.array([300.,600.,1.]);canonical=native*np.array([2.,2.,1.])
        np.testing.assert_allclose(H@native,a.native_homography_in_canonical_coordinates(H)@canonical)
    def test_aspect_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):FrameCoordinates(1920,1200)
    def test_unexpected_frame_size_is_rejected(self):
        with self.assertRaises(ValueError):FrameCoordinates(1920,1080).image(np.zeros((540,960,3),np.uint8))
    def test_legacy4K_image_has_no_resampling(self):
        im=np.zeros((2160,3840,3),np.uint8)
        self.assertIs(FrameCoordinates(3840,2160).image(im),im)


if __name__=='__main__':unittest.main()
