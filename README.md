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
