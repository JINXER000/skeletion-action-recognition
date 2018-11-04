# skeletion-action-recognition

### 工程的目标： 从视频序列中实时提取人体骨架关节点，并且分析其语义信息。

- action_recognition.py:
主函数，有模型定义
- extract_key.py: 
从json文件里得到关节点
- genDirs.py:
根据训练集生成对应名字的文件夹
- clip_save.py:
遍历原始训练集，将每个长视频分割成任意间隔的短视频片段
- performance.png:
训练过程，记录了loss 和 accuracy变化曲线


### 数据：
经过处理，每一个sample的视频序列都是（1，50，50）的numpy array.
#### 维度定义：（samples, max_timesteps, length）,其中第三个维度大小是25个关节点的x,y坐标。
- 注意：openpose输出是25*3的list,其中每个关节点输出置信度。我将其删去，只剩下坐标信息。
因为用Openpose提取关节点太耗时，于是总共选用了192段视频提取，得到（192，50，50）的tensor.
