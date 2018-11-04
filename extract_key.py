# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 14:31:05 2018

@author: Jinxe
"""

import json
import numpy as np
import os

rootDir=r'F:\CV_WS\machine_learning\openpose-1.4.0\openpose-1.4.0\out_json'
secDir=r'second01'
targetDir=r'F:\CV_WS\machine_learning\openpose-1.4.0\openpose-1.4.0\out_json'
#max_frame=60
frame_cnt=0

#output_array_full=np.zeros((None,max_frame,50))

def get_jsonLsit(rootDir,secDir):
    filePath=os.path.join(rootDir,secDir)
    jsonList=[]
    for root,dirs,files in os.walk(filePath):
        
        jsonList.extend([os.path.join(root,x) for x in files if (os.path.isfile(os.path.join(root,x))) and os.path.splitext(x)[1]=='.json'])
    return jsonList



def del_confidence(key_list):
    idx=0
    for point in key_list:
        idx=idx+1
        if idx%3==0:
            key_list[idx-1]=-1
    for point2 in key_list:
        if point2<0:
            key_list.remove(point2)
    return key_list

def pass_x_once(rootDir,secDir):
    set_num,max_frame=count_set(rootDir)
    output_x=np.zeros((1,max_frame,50))
    moveflow=get_jsonLsit(rootDir,secDir)
    if moveflow!=[]:
        output_x[0],_=extract_1frame(moveflow,max_frame)
    return output_x
        
    
def extract_1frame(jsonList,max_frame):
    output_array=np.zeros((max_frame,50))
    frame_cnt=0
    for jsonname in jsonList:
        frame_cnt=frame_cnt+1
        with open(jsonname,'r') as f:
            load_dict=json.load(f)
            people_list=load_dict["people"]
            for person in people_list:
                key_list=person["pose_keypoints_2d"]
                
                key_list=del_confidence(key_list)
                #list to ndarry
                key_array=np.array(key_list)
                try:
                    output_array[frame_cnt-1]=key_array
                except IndexError as e:
                    print("fuck!")

    return output_array,frame_cnt

def count_set(tgtDir):
    count=0
    max_json=0
    for sets in os.listdir(tgtDir):
        current_path=os.path.join(tgtDir,sets)
        if get_jsonLsit(tgtDir,sets)!=[]:
            count+=1
            current_frame_num=len(os.listdir(current_path))
            if current_frame_num>max_json:
                max_json=current_frame_num
            
    return count,max_json
def process_all_sec(tgtDir):
    set_num,max_frame=count_set(tgtDir)
    output_array_full=np.zeros((set_num,max_frame,50))
    output_yTrain=np.zeros((set_num,1),dtype=np.int)
    sec_idx=0
    for sec in os.listdir(tgtDir):
        list_1move=get_jsonLsit(tgtDir,sec)
        if list_1move!=[]:
            # gen x train
            try:
                output_array_full[sec_idx],_=extract_1frame(list_1move,max_frame)
            except  IndexError as e:
                print("fuck2")
            #gen y train
            if "walking" in sec:
                output_yTrain[sec_idx]=0
            else:
                output_yTrain[sec_idx]=1
                
            sec_idx+=1
    return output_array_full,max_frame,output_yTrain
        
def get_xTrain():
    #return extract_1frame(get_jsonLsit(rootDir,secDir))
    return process_all_sec(targetDir)

def get_yTrain():
    return np.ones((3,1),dtype=np.int)
                

                    
if __name__=='__main__':
    test_jsonList=[r'F:\CV_WS\machine_learning\openpose-1.4.0\openpose-1.4.0\out_json\second01\second01_000000000000_keypoints.json']
    
    testDir=r'F:\CV_WS\machine_learning\openpose-1.4.0\openpose-1.4.0\out_json'
    
    jsonList=get_jsonLsit(rootDir,secDir)
    X_train=process_all_sec(testDir)
    #output_list,frame_cnt=  extract_1frame(jsonList)
    