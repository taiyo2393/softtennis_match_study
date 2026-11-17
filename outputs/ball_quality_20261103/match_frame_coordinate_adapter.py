"""Explicit image-coordinate conversion; never camera calibration or detail gain."""
import cv2
import numpy as np


class FrameCoordinates:
    def __init__(self,width,height,canonical_width=3840,canonical_height=2160):
        if min(width,height,canonical_width,canonical_height)<=0:raise ValueError('Dimensions must be positive.')
        if width*canonical_height!=height*canonical_width:raise ValueError('Aspect mismatch: do not silently stretch or crop.')
        self.width,self.height=width,height
        self.canonical_width,self.canonical_height=canonical_width,canonical_height
        self.scale=canonical_width/width

    def image(self,image):
        if image.shape[:2]!=(self.height,self.width):raise ValueError('Unexpected native image dimensions.')
        if self.scale==1:return image
        return cv2.resize(image,(self.canonical_width,self.canonical_height),interpolation=cv2.INTER_LINEAR if self.scale>1 else cv2.INTER_AREA)

    def row(self,row,to_canonical=True):
        result=dict(row);factor=self.scale if to_canonical else 1/self.scale
        for key in ['x1','y1','x2','y2','bottom_center_x','bottom_center_y']:
            if key in row and row[key] is not None and row[key]!='':result[key]=float(row[key])*factor
        return result

    def keypoints(self,keypoints,to_canonical=True):
        result=np.asarray(keypoints).copy()
        result[...,:2]*=self.scale if to_canonical else 1/self.scale
        return result

    def native_homography_in_canonical_coordinates(self,native_homography):
        # Caller must supply a NEW valid calibration for the current camera.
        # This changes its input coordinate system, not the underlying camera.
        value=np.asarray(native_homography,dtype=float)
        if value.shape!=(3,3) or not np.isfinite(value).all():raise ValueError('Finite3x3 homography required.')
        return value@np.diag([1/self.scale,1/self.scale,1.])
