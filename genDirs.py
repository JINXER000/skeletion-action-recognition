# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 20:58:13 2018

@author: Jinxe
"""
import os
def gen_dir(filePath,genPath):
        for root,dirs,files in os.walk(filePath):
            for x in files :
                if (os.path.isfile(os.path.join(root,x))) and os.path.splitext(x)[1]=='.avi':
                    genDir=os.path.join(genPath,os.path.splitext(x)[0])
                    os.makedirs(genDir)
                    
if __name__=='__main__':
    filePath=r'F:\CV_WS\machine_learning\openpose-1.4.0\openpose-1.4.0\out_json\py_scripts\video_frags'
    genPath=r'F:\CV_WS\machine_learning\openpose-1.4.0\openpose-1.4.0\out_json'
    gen_dir(filePath,genPath)