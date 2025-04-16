import numpy as np

# 读取训练和测试数据集
train = list(open("Anomaly_Train.txt"))
test = list(open("Anomaly_Test.txt"))

# 处理训练集文件
with open('ucf_x3d_train.txt', 'w+') as f:
    normal_files = []  # 存储正常文件的列表
    for file in train:
        if "Normal" in file:
            # 如果文件中包含"Normal"，则认为是正常文件
            normal_files.append(file)
        else:
            # 如果文件不包含"Normal"，将文件名转换为'.npy'格式并写入
            newline = 'X3D_Videos/' + file[:-4] + 'npy\n'
            f.write(newline)
    
    # 处理正常文件，将其转换为'.npy'格式并写入
    for file in normal_files:
        newline = 'X3D_Videos/' + file[:-4] + 'npy\n'
        f.write(newline)

# 处理测试集文件
with open('ucf_x3d_test.txt', 'w+') as f:
    normal_files = []  # 存储正常文件的列表
    for file in test:
        if "Normal" in file:
            # 如果文件中包含"Normal"，则认为是正常文件
            normal_files.append(file)
        else:
            # 如果文件不包含"Normal"，将文件名转换为'.npy'格式并写入
            newline = 'X3D_Videos/' + file[:-4] + 'npy\n'
            f.write(newline)
    
    # 处理正常文件，将其转换为'.npy'格式并写入
    for file in normal_files:
        newline = 'X3D_Videos/' + file[:-4] + 'npy\n'
        f.write(newline)
