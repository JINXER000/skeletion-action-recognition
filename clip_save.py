# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 10:55:47 2018

@author: Jinxe
"""

import cv2
import os
import numpy as np

def clipVideo(max_frame,fileName):
    videoCapture=cv2.VideoCapture(fileName)
    fourcc=cv2.VideoWriter_fourcc(*'XVID')
    _, image = videoCapture.read()
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    size = (int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)), 
            int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    ret=1
    
    mp4_cnt=0
    while(1):
        
        for i in range(max_frame):
            ret,srcimg=videoCapture.read()
            # try:
            
            #     cv2.imshow("raw img",srcimg)
            # except:
            #     print("fuck")
            out_name='video_frags/'+fileName.split('.')[0]+str(mp4_cnt)+'.avi'
            videoWriter=cv2.VideoWriter(out_name,fourcc, fps, size)
            videoWriter.write(srcimg)
            
        cv2.waitKey(int(max_frame/fps))
        mp4_cnt+=1
        
        

        
if __name__=='__main__': 
    test_file=r"person01_boxing_d1_uncomp.avi"
    max_frame=1000
    clipVideo(max_frame,test_file)           