#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /data/zhangruochi/projects/fewshot_admet/dataset/samples.py
# Project: /home/richard/projects/fsadmet/model
# Created Date: Tuesday, June 28th 2022, 6:41:53 pm
# Author: Ruochi Zhang
# Email: zrc720@gmail.com
# -----
# Last Modified: Wed Jun 05 2024
# Modified By: Ruochi Zhang
# -----
# Copyright (c) 2022 Bodkin World Domination Enterprises
# 
# MIT License
# 
# Copyright (c) 2022 HILAB
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -----
###
# from torch_geometric.loader import DataLoader
# import torch
# import random
# from torch.utils.data import Dataset, Subset
# import torch_geometric
# from torch_geometric.data import Data
# from torch_geometric.data import InMemoryDataset

# # def obtain_distr_list(dataset):
#     # if dataset == "sider":
#     #     return [[684,743],[431,996],[1405,22],[551,876],[276,1151],[430,997],[129,1298],[1176,251],[403,1024],[700,727],[1051,376],[135,1292],[1104,323],[1214,213],[319,1108],[542,885],[109,1318],[1174,253],[421,1006],[367,1060],[411,1016],[516,911],[1302,125],[768,659],[439,988],[123,1304],[481,946]]
#     # elif dataset == "tox21":
#     #     return [[6956,309],[6521,237],[5781,768],[5521,300],[5400,793],[6605,350],[6264,186],[4890,942],[6808,264],[6095,372],[4892,918],[6351,423]]
#     # elif dataset == "muv":
#     #     return [[14814,27],[14705,29],[14698,30],[14593,30],[14873,29],[14572,29],[14614,30],[14383,28],[14807,29],[14654,28],[14662,29],[14615,29],[14637,30],[14681,30],[14622,29],[14745,29],[14722,24]]
#     # elif dataset == "toxcast":
#     #     return [[1293, 438], [1441, 290], [864, 170], [995, 39], [794, 240], [738, 296], [591, 443], [977, 57], [948, 86], [960, 59], [910, 109], [908, 126], [1010, 24], [930, 89], [947, 72], [281, 22], [849, 185], [889, 130], [822, 212], [740, 279], [979, 55], [994, 40], [1018, 16], [797, 237], [788, 246], [286, 17], [967, 67], [935, 99], [842, 192], [828, 206], [262, 41], [257, 46], [252, 51], [267, 36], [251, 52], [248, 55], [247, 56], [251, 52], [251, 52], [262, 41], [263, 40], [283, 20], [274, 29], [286, 17], [284, 19], [3333, 79], [2796, 616], [3198, 214], [3379, 33], [3400, 12], [3382, 30], [3184, 228], [2975, 437], [3321, 91], [3032, 380], [3341, 71], [3386, 26], [3279, 133], [2875, 537], [3363, 49], [3061, 351], [3341, 71], [3172, 240], [2716, 696], [3357, 55], [3278, 134], [3094, 318], [3287, 125], [3390, 22], [2913, 499], [3207, 205], [2543, 869], [3386, 26], [3388, 24], [3402, 10], [2685, 727], [3081, 331], [3340, 72], [3195, 217], [3395, 17], [3320, 92], [3353, 59], [3350, 62], [3256, 156], [3370, 42], [3068, 76], [3310, 102], [3376, 36], [3380, 32], [3231, 181], [3271, 141], [3367, 45], [3395, 17], [3363, 49], [3193, 219], [3036, 376], [3388, 24], [3373, 39], [3293, 119], [3356, 56], [3367, 45], [2998, 414], [3078, 334], [3330, 82], [2947, 465], [3397, 15], [3359, 53], [3319, 93], [3397, 15], [3346, 66], [2696, 716], [3400, 12], [3338, 74], [3356, 56], [3386, 26], [3364, 48], [3370, 42], [3363, 49], [3392, 20], [3401, 11], [3299, 113], [3371, 41], [3372, 40], [3233, 179], [3365, 47], [3146, 266], [3142, 270], [3282, 130], [3265, 147], [3319, 93], [3367, 45], [2123, 1289], [3392, 20], [3208, 204], [3386, 26], [2867, 545], [3392, 20], [3026, 386], [3385, 27], [3184, 228], [3351, 61], [2484, 928], [3330, 82], [2887, 525], [3090, 322], [1769, 1643], [3400, 12], [2445, 967], [2963, 449], [3176, 236], [3344, 68], [3285, 127], [3397, 15], [3357, 55], [3364, 48], [3393, 19], [3041, 371], [3368, 44], [3393, 19], [3387, 25], [3399, 13], [3314, 98], [3324, 88], [2893, 519], [3379, 33], [2887, 525], [3323, 89], [3381, 31], [3389, 23], [3198, 214], [3388, 24], [3071, 341], [3357, 55], [3300, 112], [3394, 18], [3186, 226], [2958, 454], [3382, 30], [3299, 113], [3285, 127], [3384, 28], [3311, 101], [3403, 9], [2486, 926], [3398, 14], [3373, 39], [2648, 245], [3393, 19], [2960, 452], [3083, 329], [3334, 78], [1043, 396], [889, 550], [1240, 199], [1114, 325], [1062, 377], [1254, 185], [842, 597], [964, 475], [1383, 56], [1306, 133], [1124, 315], [1385, 54], [1090, 349], [1006, 433], [1026, 413], [1006, 433], [1058, 381], [1041, 398], [1423, 16], [1051, 388], [1018, 421], [1229, 210], [1098, 341], [1424, 15], [1056, 383], [1174, 265], [1060, 379], [1253, 186], [1253, 186], [1312, 127], [1150, 289], [1235, 204], [1215, 224], [1198, 241], [1244, 195], [1406, 33], [1204, 235], [1154, 285], [1235, 204], [1396, 43], [1299, 140], [1281, 158], [1361, 78], [1231, 208], [1413, 26], [1180, 259], [1423, 16], [1287, 152], [998, 441], [1417, 22], [1267, 172], [1409, 30], [1193, 246], [1371, 68], [1191, 248], [1223, 216], [1160, 279], [1407, 32], [1197, 242], [1422, 17], [1218, 221], [1147, 292], [1121, 318], [1420, 19], [1186, 253], [1419, 20], [1053, 386], [1211, 228], [1151, 288], [1119, 320], [1177, 262], [1019, 420], [1138, 301], [1423, 16], [1134, 305], [1423, 16], [1124, 315], [1414, 25], [1119, 320], [1047, 392], [1146, 293], [1349, 90], [1070, 369], [1151, 288], [1368, 71], [1208, 231], [1390, 49], [1003, 436], [1000, 439], [998, 441], [1040, 399], [1034, 405], [1398, 41], [1096, 343], [1402, 37], [1096, 343], [1212, 227], [1123, 316], [1367, 72], [877, 562], [1079, 360], [1006, 433], [1347, 92], [1382, 57], [1252, 187], [1023, 416], [1027, 412], [1149, 290], [1178, 261], [1380, 59], [1049, 390], [817, 622], [1112, 327], [1176, 263], [1032, 407], [300, 202], [318, 184], [369, 133], [365, 131], [427, 69], [470, 30], [428, 72], [459, 43], [436, 66], [411, 51], [353, 147], [387, 113], [351, 118], [358, 142], [283, 17], [279, 21], [176, 120], [186, 109], [201, 101], [169, 133], [147, 153], [221, 81], [128, 171], [139, 161], [121, 181], [178, 114], [178, 116], [254, 42], [272, 28], [277, 22], [261, 39], [252, 50], [236, 64], [173, 200], [276, 97], [143, 175], [66, 307], [22, 31], [221, 482], [168, 71], [105, 70], [39, 134], [86, 27], [35, 101], [76, 301], [38, 187], [37, 80], [75, 85], [49, 28], [23, 31], [74, 68], [90, 21], [72, 23], [80, 90], [42, 37], [99, 31], [43, 60], [81, 80], [59, 54], [136, 29], [196, 24], [55, 44], [37, 45], [55, 35], [70, 34], [72, 21], [58, 39], [53, 26], [80, 58], [113, 67], [92, 20], [65, 31], [63, 24], [54, 25], [51, 24], [76, 32], [29, 38], [88, 26], [69, 29], [42, 21], [130, 24], [56, 84], [42, 61], [50, 49], [56, 39], [31, 84], [42, 64], [57, 71], [76, 56], [52, 54], [74, 38], [24, 31], [50, 85], [43, 77], [36, 53], [37, 28], [45, 57], [55, 91], [63, 46], [66, 89], [35, 65], [40, 120], [46, 21], [34, 84], [20, 66], [30, 61], [31, 81], [38, 57], [38, 40], [61, 25], [32, 98], [53, 72], [21, 57], [33, 57], [49, 22], [26, 57], [43, 75], [32, 70], [49, 81], [85, 79], [47, 60], [75, 114], [35, 60], [41, 70], [43, 29], [44, 48], [41, 51], [40, 53], [25, 53], [42, 23], [66, 46], [57, 28], [57, 72], [57, 65], [37, 33], [915, 27], [25, 30], [42, 57], [26, 77], [51, 40], [31, 71], [35, 54], [41, 117], [42, 25], [43, 23], [24, 26], [37, 25], [54, 30], [133, 215], [116, 217], [927, 127], [110, 75], [98, 206], [116, 112], [194, 83], [900, 228], [133, 31], [198, 59], [120, 225], [304, 72], [602, 178], [196, 85], [405, 109], [231, 29], [145, 21], [168, 55], [742, 186], [139, 131], [77, 20], [38, 107], [50, 123], [26, 51], [50, 193], [69, 160], [64, 39], [39, 39], [52, 61], [53, 49], [1635, 137], [1629, 111], [1532, 201], [1623, 125], [1575, 99], [1544, 211], [1478, 190], [1543, 201], [1497, 169], [1596, 175], [1619, 139], [1424, 311], [1549, 133], [1560, 198], [1657, 80], [6835, 352], [6564, 623], [5583, 1604], [7118, 69], [7926, 5], [7562, 369], [7540, 391], [7908, 23], [7746, 185], [6773, 1158], [7351, 580], [7565, 366], [7100, 831], [6034, 1153], [7141, 790], [6674, 1257], [7900, 31], [7898, 33], [7899, 32], [7926, 5], [7901, 30], [7927, 4], [5151, 120], [7653, 278], [7482, 449], [7480, 451], [7650, 281], [7694, 237], [6919, 1012], [7750, 181], [6691, 1240], [7234, 697], [7110, 77], [7094, 93], [6871, 316], [6971, 216], [6843, 344], [6917, 270], [7020, 167], [6997, 190], [6243, 944], [6871, 316], [7620, 311], [7721, 210], [7448, 483], [7413, 518], [7492, 439], [7550, 381], [6909, 278], [6830, 357], [6592, 595], [7035, 152], [4425, 846], [5163, 108], [4982, 289], [6908, 279], [7100, 87], [6961, 226], [6755, 432], [6551, 636], [7084, 103], [7184, 3], [7017, 170], [7010, 177], [6761, 426], [7171, 16], [7926, 5], [7716, 215], [7596, 335], [7175, 12], [6655, 532], [7045, 142], [7912, 19], [6186, 1745], [6572, 615], [7027, 160], [7140, 47], [7134, 53], [6890, 297], [6926, 261], [7380, 551], [7479, 452], [7136, 795], [7556, 375], [7423, 508], [7149, 782], [6919, 1012], [7152, 779], [7269, 662], [7239, 692], [6991, 940], [7516, 415], [7292, 639], [7379, 552], [7001, 930], [7279, 652], [7596, 335], [7306, 625], [7066, 865], [7622, 309], [910, 111], [840, 194], [962, 59], [975, 46], [1006, 15], [945, 76], [911, 110], [914, 120], [982, 39], [903, 118], [969, 52], [979, 42], [914, 107], [985, 36], [991, 30], [966, 55], [942, 79], [895, 126]]

# def obtain_distr_list(dataset, processed_data):
#     if dataset == "sider":
#         # 原始分布列表
#         original_distr_list = [
#             [684, 743], [431, 996], [1405, 22], [551, 876], [276, 1151],
#             [430, 997], [129, 1298], [1176, 251], [403, 1024], [700, 727],
#             [1051, 376], [135, 1292], [1104, 323], [1214, 213], [319, 1108],
#             [542, 885], [109, 1318], [1174, 253], [421, 1006], [367, 1060],
#             [411, 1016], [516, 911], [1302, 125], [768, 659], [439, 988],
#             [123, 1304], [481, 946]
#         ]
        
#         # 重新计算分布列表
#         distr_list = []
#         current_index = 0
#         for start, end in original_distr_list:
#             # 跳过没有边的分子
#             filtered_end = current_index + count_valid_molecules(processed_data[current_index:end])
#             distr_list.append([current_index, filtered_end])
#             current_index = filtered_end
        
#         return distr_list
    
#     elif dataset == "tox21":
#         # 原始分布列表
#         original_distr_list = [
#             [6956, 309], [6521, 237], [5781, 768], [5521, 300], [5400, 793],
#             [6605, 350], [6264, 186], [4890, 942], [6808, 264], [6095, 372],
#             [4892, 918], [6351, 423]
#         ]
        
#         # 重新计算分布列表
#         distr_list = []
#         current_index = 0
#         for start, end in original_distr_list:
#             # 跳过没有边的分子
#             filtered_end = current_index + count_valid_molecules(processed_data[current_index:end])
#             distr_list.append([current_index, filtered_end])
#             current_index = filtered_end

#         return distr_list
#     elif dataset == "muv":
#         original_distr_list = [
#             [14814, 27], [14705, 29], [14698, 30], [14593, 30],
#             [14873, 29], [14572, 29], [14614, 30], [14383, 28],
#             [14807, 29], [14654, 28], [14662, 29], [14615, 29],
#             [14637, 30], [14681, 30], [14622, 29], [14745, 29],
#             [14722, 24]
#         ]
#         # 重新计算分布列表
#         distr_list = []
#         current_index = 0
#         for start, end in original_distr_list:
#             # 跳过没有边的分子
#             filtered_end = current_index + count_valid_molecules(processed_data[current_index:end])
#             distr_list.append([current_index, filtered_end])
#             current_index = filtered_end
#             # print(f"Task start: {current_index}, end: {filtered_end}")  # 添加打印语句
#         return distr_list
#     else:
#         raise ValueError(f"Unsupported dataset: {dataset}")

# def count_valid_molecules(data_subset):
#     # 计算子集中有效的分子数量
#     valid_count = 0
#     for mol in data_subset:
#         print(f'num_edges:', mol.num_edges)
#         if mol.num_edges > 0:
#             valid_count += 1
#     return valid_count


# def sample_datasets(data, dataset, task, n_way, m_support, n_query):
#     distri_list = obtain_distr_list(dataset, data)
#     print(f"distri_list: {distri_list}, data length: {len(data)}")
    

#     # 确保 task 的值在有效范围内
#     # if task >= len(distri_list):
#     #     raise ValueError(f"Task {task} is out of bounds for dataset {dataset}. Valid tasks are 0 to {len(distri_list) - 1}")
#     # 确保采样范围在数据集内
#     start, end = distri_list[task]
#     if start >= len(data):
#         raise ValueError(f"Task {task} start index {start} is out of bounds for data length {len(data)}")
#     if end > len(data):
#         raise ValueError(f"Task {task} end index {end} is out of bounds for data length {len(data)}")

#     # 从数据集中采样支持集
#     available_indices = list(range(0, start)) + list(range(start, end))
#     if len(available_indices) < m_support:
#         # 如果可用的索引不够，使用所有可用的索引
#         support_list = available_indices
#     else:
#         support_list = random.sample(available_indices, m_support)

#     # 从剩余的分子中采样查询集
#     l = [i for i in range(0, len(data)) if i not in support_list]
#     query_list = random.sample(l, n_query)

# # 检查 data 是否是一个 InMemoryDataset 类型的对象。
#     if isinstance(data, InMemoryDataset):
#         support_dataset = data[torch.tensor(support_list)]
#         query_dataset = data[torch.tensor(query_list)]
#     elif isinstance(data, torch.utils.data.Dataset):
#         support_dataset = Subset(data, support_list)
#         query_dataset = Subset(data, query_list)
    

#     return support_dataset, query_dataset


# import random
# from torch.utils.data import Subset

# def sample_test_datasets(data, dataset, task, n_way, m_support, n_query):
#     distri_list = obtain_distr_list(dataset, data)
    
#     # 确保 task 的值在有效范围内
#     if task >= len(distri_list):
#         raise ValueError(f"Task {task} is out of bounds for dataset {dataset}. Valid tasks are 0 to {len(distri_list) - 1}")
#     # 确保采样范围在数据集内
#     start, end = distri_list[task]
#     if start >= len(data):
#         raise ValueError(f"Task {task} start index {start} is out of bounds for data length {len(data)}")
#     if end > len(data):
#         raise ValueError(f"Task {task} end index {end} is out of bounds for data length {len(data)}")

#     # 从数据集中采样支持集
#     # 先从开始之前的部分采样，如果不够再从开始到结束之间采样
#     support_list_first_part = random.sample(range(0, start), min(m_support, start))
#     remaining_support_needed = m_support - len(support_list_first_part)
#     support_list_second_part = random.sample(range(start, end), min(remaining_support_needed, end-start))

#     support_list = support_list_first_part + support_list_second_part
#     random.shuffle(support_list)  # 打乱支持集样本的顺序

#     # 从剩余的样本中采样查询集
#     remaining_indices = [i for i in range(len(data)) if i not in support_list]
#     random.shuffle(remaining_indices)
#     query_list = remaining_indices[:n_query]  # 取前n_query个作为查询集

#     if isinstance(data, torch_geometric.data.InMemoryDataset):
#         support_dataset = data[torch.tensor(support_list)]
#         query_dataset = data[torch.tensor(query_list)]
#     elif isinstance(data, torch.utils.data.Dataset):
#         support_dataset = Subset(data, support_list)
#         query_dataset = Subset(data, query_list)
    
#     return support_dataset, query_dataset


import torch
import random
from torch.utils.data import Dataset, Subset
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
import random
from torch.utils.data import Subset
from torch_geometric.data import InMemoryDataset
import random
from torch.utils.data import Subset
from torch_geometric.data import InMemoryDataset

def obtain_distr_list(dataset,data = None):
    if dataset == "sider":
        return  [[655, 741], [418, 978], [1374, 22], [530, 866], [263, 1133], [411, 985], [120, 1276], [1146, 250], [384, 1012], [678, 718], [1029, 367], [129, 1267], [1076, 320], [1185, 211], [306, 1090], [527, 869], [99, 1297], [1144, 252], [410, 986], [351, 1045], [392, 1004], [496, 900], [1273, 123], [745, 651], [423, 973], [109, 1287], [465, 931]]
    elif dataset == "tox21":
        return [[6937, 309], [6505, 236], [5764, 766], [5506, 299], [5383, 793], [6588, 349], [6248, 185], [4878, 939], [6792, 263], [6079, 369], [4879, 917], [6335, 422]]
    elif dataset == "muv":
        return [
    [14814, 27], [14705, 29], [14698, 30], [14593, 30],
    [14873, 29], [14572, 29], [14614, 30], [14383, 28],
    [14807, 29], [14654, 28], [14662, 29], [14615, 29],
    [14637, 30], [14681, 30], [14622, 29], [14745, 29],
    [14722, 24]
    ]
    elif dataset == "toxcast":
        return [[1293, 438], [1441, 290], [864, 170], [995, 39], [794, 240], [738, 296], [591, 443], [977, 57], [948, 86], [960, 59], [910, 109], [908, 126], [1010, 24], [930, 89], [947, 72], [281, 22], [849, 185], [889, 130], [822, 212], [740, 279], [979, 55], [994, 40], [1018, 16], [797, 237], [788, 246], [286, 17], [967, 67], [935, 99], [842, 192], [828, 206], [262, 41], [257, 46], [252, 51], [267, 36], [251, 52], [248, 55], [247, 56], [251, 52], [251, 52], [262, 41], [263, 40], [283, 20], [274, 29], [286, 17], [284, 19], [3333, 79], [2796, 616], [3198, 214], [3379, 33], [3400, 12], [3382, 30], [3184, 228], [2975, 437], [3321, 91], [3032, 380], [3341, 71], [3386, 26], [3279, 133], [2875, 537], [3363, 49], [3061, 351], [3341, 71], [3172, 240], [2716, 696], [3357, 55], [3278, 134], [3094, 318], [3287, 125], [3390, 22], [2913, 499], [3207, 205], [2543, 869], [3386, 26], [3388, 24], [3402, 10], [2685, 727], [3081, 331], [3340, 72], [3195, 217], [3395, 17], [3320, 92], [3353, 59], [3350, 62], [3256, 156], [3370, 42], [3068, 76], [3310, 102], [3376, 36], [3380, 32], [3231, 181], [3271, 141], [3367, 45], [3395, 17], [3363, 49], [3193, 219], [3036, 376], [3388, 24], [3373, 39], [3293, 119], [3356, 56], [3367, 45], [2998, 414], [3078, 334], [3330, 82], [2947, 465], [3397, 15], [3359, 53], [3319, 93], [3397, 15], [3346, 66], [2696, 716], [3400, 12], [3338, 74], [3356, 56], [3386, 26], [3364, 48], [3370, 42], [3363, 49], [3392, 20], [3401, 11], [3299, 113], [3371, 41], [3372, 40], [3233, 179], [3365, 47], [3146, 266], [3142, 270], [3282, 130], [3265, 147], [3319, 93], [3367, 45], [2123, 1289], [3392, 20], [3208, 204], [3386, 26], [2867, 545], [3392, 20], [3026, 386], [3385, 27], [3184, 228], [3351, 61], [2484, 928], [3330, 82], [2887, 525], [3090, 322], [1769, 1643], [3400, 12], [2445, 967], [2963, 449], [3176, 236], [3344, 68], [3285, 127], [3397, 15], [3357, 55], [3364, 48], [3393, 19], [3041, 371], [3368, 44], [3393, 19], [3387, 25], [3399, 13], [3314, 98], [3324, 88], [2893, 519], [3379, 33], [2887, 525], [3323, 89], [3381, 31], [3389, 23], [3198, 214], [3388, 24], [3071, 341], [3357, 55], [3300, 112], [3394, 18], [3186, 226], [2958, 454], [3382, 30], [3299, 113], [3285, 127], [3384, 28], [3311, 101], [3403, 9], [2486, 926], [3398, 14], [3373, 39], [2648, 245], [3393, 19], [2960, 452], [3083, 329], [3334, 78], [1043, 396], [889, 550], [1240, 199], [1114, 325], [1062, 377], [1254, 185], [842, 597], [964, 475], [1383, 56], [1306, 133], [1124, 315], [1385, 54], [1090, 349], [1006, 433], [1026, 413], [1006, 433], [1058, 381], [1041, 398], [1423, 16], [1051, 388], [1018, 421], [1229, 210], [1098, 341], [1424, 15], [1056, 383], [1174, 265], [1060, 379], [1253, 186], [1253, 186], [1312, 127], [1150, 289], [1235, 204], [1215, 224], [1198, 241], [1244, 195], [1406, 33], [1204, 235], [1154, 285], [1235, 204], [1396, 43], [1299, 140], [1281, 158], [1361, 78], [1231, 208], [1413, 26], [1180, 259], [1423, 16], [1287, 152], [998, 441], [1417, 22], [1267, 172], [1409, 30], [1193, 246], [1371, 68], [1191, 248], [1223, 216], [1160, 279], [1407, 32], [1197, 242], [1422, 17], [1218, 221], [1147, 292], [1121, 318], [1420, 19], [1186, 253], [1419, 20], [1053, 386], [1211, 228], [1151, 288], [1119, 320], [1177, 262], [1019, 420], [1138, 301], [1423, 16], [1134, 305], [1423, 16], [1124, 315], [1414, 25], [1119, 320], [1047, 392], [1146, 293], [1349, 90], [1070, 369], [1151, 288], [1368, 71], [1208, 231], [1390, 49], [1003, 436], [1000, 439], [998, 441], [1040, 399], [1034, 405], [1398, 41], [1096, 343], [1402, 37], [1096, 343], [1212, 227], [1123, 316], [1367, 72], [877, 562], [1079, 360], [1006, 433], [1347, 92], [1382, 57], [1252, 187], [1023, 416], [1027, 412], [1149, 290], [1178, 261], [1380, 59], [1049, 390], [817, 622], [1112, 327], [1176, 263], [1032, 407], [300, 202], [318, 184], [369, 133], [365, 131], [427, 69], [470, 30], [428, 72], [459, 43], [436, 66], [411, 51], [353, 147], [387, 113], [351, 118], [358, 142], [283, 17], [279, 21], [176, 120], [186, 109], [201, 101], [169, 133], [147, 153], [221, 81], [128, 171], [139, 161], [121, 181], [178, 114], [178, 116], [254, 42], [272, 28], [277, 22], [261, 39], [252, 50], [236, 64], [173, 200], [276, 97], [143, 175], [66, 307], [22, 31], [221, 482], [168, 71], [105, 70], [39, 134], [86, 27], [35, 101], [76, 301], [38, 187], [37, 80], [75, 85], [49, 28], [23, 31], [74, 68], [90, 21], [72, 23], [80, 90], [42, 37], [99, 31], [43, 60], [81, 80], [59, 54], [136, 29], [196, 24], [55, 44], [37, 45], [55, 35], [70, 34], [72, 21], [58, 39], [53, 26], [80, 58], [113, 67], [92, 20], [65, 31], [63, 24], [54, 25], [51, 24], [76, 32], [29, 38], [88, 26], [69, 29], [42, 21], [130, 24], [56, 84], [42, 61], [50, 49], [56, 39], [31, 84], [42, 64], [57, 71], [76, 56], [52, 54], [74, 38], [24, 31], [50, 85], [43, 77], [36, 53], [37, 28], [45, 57], [55, 91], [63, 46], [66, 89], [35, 65], [40, 120], [46, 21], [34, 84], [20, 66], [30, 61], [31, 81], [38, 57], [38, 40], [61, 25], [32, 98], [53, 72], [21, 57], [33, 57], [49, 22], [26, 57], [43, 75], [32, 70], [49, 81], [85, 79], [47, 60], [75, 114], [35, 60], [41, 70], [43, 29], [44, 48], [41, 51], [40, 53], [25, 53], [42, 23], [66, 46], [57, 28], [57, 72], [57, 65], [37, 33], [915, 27], [25, 30], [42, 57], [26, 77], [51, 40], [31, 71], [35, 54], [41, 117], [42, 25], [43, 23], [24, 26], [37, 25], [54, 30], [133, 215], [116, 217], [927, 127], [110, 75], [98, 206], [116, 112], [194, 83], [900, 228], [133, 31], [198, 59], [120, 225], [304, 72], [602, 178], [196, 85], [405, 109], [231, 29], [145, 21], [168, 55], [742, 186], [139, 131], [77, 20], [38, 107], [50, 123], [26, 51], [50, 193], [69, 160], [64, 39], [39, 39], [52, 61], [53, 49], [1635, 137], [1629, 111], [1532, 201], [1623, 125], [1575, 99], [1544, 211], [1478, 190], [1543, 201], [1497, 169], [1596, 175], [1619, 139], [1424, 311], [1549, 133], [1560, 198], [1657, 80], [6835, 352], [6564, 623], [5583, 1604], [7118, 69], [7926, 5], [7562, 369], [7540, 391], [7908, 23], [7746, 185], [6773, 1158], [7351, 580], [7565, 366], [7100, 831], [6034, 1153], [7141, 790], [6674, 1257], [7900, 31], [7898, 33], [7899, 32], [7926, 5], [7901, 30], [7927, 4], [5151, 120], [7653, 278], [7482, 449], [7480, 451], [7650, 281], [7694, 237], [6919, 1012], [7750, 181], [6691, 1240], [7234, 697], [7110, 77], [7094, 93], [6871, 316], [6971, 216], [6843, 344], [6917, 270], [7020, 167], [6997, 190], [6243, 944], [6871, 316], [7620, 311], [7721, 210], [7448, 483], [7413, 518], [7492, 439], [7550, 381], [6909, 278], [6830, 357], [6592, 595], [7035, 152], [4425, 846], [5163, 108], [4982, 289], [6908, 279], [7100, 87], [6961, 226], [6755, 432], [6551, 636], [7084, 103], [7184, 3], [7017, 170], [7010, 177], [6761, 426], [7171, 16], [7926, 5], [7716, 215], [7596, 335], [7175, 12], [6655, 532], [7045, 142], [7912, 19], [6186, 1745], [6572, 615], [7027, 160], [7140, 47], [7134, 53], [6890, 297], [6926, 261], [7380, 551], [7479, 452], [7136, 795], [7556, 375], [7423, 508], [7149, 782], [6919, 1012], [7152, 779], [7269, 662], [7239, 692], [6991, 940], [7516, 415], [7292, 639], [7379, 552], [7001, 930], [7279, 652], [7596, 335], [7306, 625], [7066, 865], [7622, 309], [910, 111], [840, 194], [962, 59], [975, 46], [1006, 15], [945, 76], [911, 110], [914, 120], [982, 39], [903, 118], [969, 52], [979, 42], [914, 107], [985, 36], [991, 30], [966, 55], [942, 79], [895, 126]]





import torch
from torch.utils.data import Subset, Dataset, DataLoader
from torch_geometric.data import InMemoryDataset

# def sample_datasets(data, dataset, task, n_way, m_support, n_query):
#     distri_list = obtain_distr_list(dataset, data)
    
#     if distri_list is None:
#         raise ValueError("distri_list is None, check the obtain_distr_list function")
    
#     # 确保 task 的值在有效范围内
#     if task >= len(distri_list):
#         raise ValueError(f"Task {task} is out of bounds for dataset {dataset}. Valid tasks are 0 to {len(distri_list) - 1}")
    
#     # 确保采样范围在数据集内
#     start, end = distri_list[task]
#     end = start + end  # 计算正确的结束索引
#     if start >= len(data):
#         raise ValueError(f"Task {task} start index {start} is out of bounds for data length {len(data)}")
#     if end > len(data):
#         raise ValueError(f"Task {task} end index {end} is out of bounds for data length {len(data)}")

#     # 获取任务的数据子集
#     task_subset = list(range(start, end))
#     task_subset_length = len(task_subset)

#     # 检查数据子集长度是否足够
#     if task_subset_length == 0:
#         raise ValueError(f"Task {task} has no valid molecules in the subset.")
#     if task_subset_length < m_support:
#         raise ValueError(f"Task {task} has only {task_subset_length} valid molecules, but requires {m_support} support samples")

#     # 从任务的数据子集中采样支持集
#     support_list = random.sample(task_subset, m_support)

#     # 从剩余的分子中采样查询集
#     remaining_indices = [i for i in range(len(data)) if i not in support_list]

#     # 分离正负样本
#     positive_indices = [i for i in remaining_indices if data[i].y.item() == 1]
#     negative_indices = [i for i in remaining_indices if data[i].y.item() == 0]

#     # 确保有足够的正负样本
#     if len(positive_indices) < n_query // 2 or len(negative_indices) < n_query // 2:
#         # 如果正样本或负样本不足，调整 n_query 以适应实际情况
#         max_possible_query = min(len(positive_indices) + len(negative_indices), n_query)
#         n_query = max_possible_query
#         positive_query_count = min(len(positive_indices), n_query // 2)
#         negative_query_count = n_query - positive_query_count

#         positive_query = random.sample(positive_indices, positive_query_count)
#         negative_query = random.sample(negative_indices, negative_query_count)
#     else:
#         positive_query = random.sample(positive_indices, n_query // 2)
#         negative_query = random.sample(negative_indices, n_query // 2)

#     # 合并正负样本查询集
#     query_list = positive_query + negative_query
#     random.shuffle(query_list)

#     # 检查 data 是否是一个 InMemoryDataset 类型的对象。
#     if isinstance(data, InMemoryDataset):
#         support_dataset = data[torch.tensor(support_list)]
#         query_dataset = data[torch.tensor(query_list)]
#     elif isinstance(data, Dataset):
#         support_dataset = Subset(data, support_list)
#         query_dataset = Subset(data, query_list)
    
#     return support_dataset, query_dataset

# # def sample_test_datasets(data, dataset, task, n_way, m_support, n_query):
#     distri_list = obtain_distr_list(dataset, data)
    
#     if distri_list is None:
#         raise ValueError("distri_list is None, check the obtain_distr_list function")
    
#     # 确保 task 的值在有效范围内
#     if task >= len(distri_list):
#         raise ValueError(f"Task {task} is out of bounds for dataset {dataset}. Valid tasks are 0 to {len(distri_list) - 1}")
    
#     # 确保采样范围在数据集内
#     start, end = distri_list[task]
#     end = start + end  # 计算正确的结束索引
#     if start >= len(data):
#         raise ValueError(f"Task {task} start index {start} is out of bounds for data length {len(data)}")
#     if end > len(data):
#         raise ValueError(f"Task {task} end index {end} is out of bounds for data length {len(data)}")

#     # 获取任务的数据子集
#     task_subset = list(range(start, end))
#     task_subset_length = len(task_subset)

#     # 检查数据子集长度是否足够
#     if task_subset_length < m_support:
#         raise ValueError(f"Task {task} has only {task_subset_length} valid molecules, but requires {m_support} support samples")

#     # 从任务的数据子集中采样支持集
#     support_list = random.sample(task_subset, m_support)

#     # 从剩余的分子中采样查询集
#     remaining_indices = [i for i in range(len(data)) if i not in support_list]

#     # 分离正负样本
#     positive_indices = [i for i in remaining_indices if data[i].y.item() == 1]
#     negative_indices = [i for i in remaining_indices if data[i].y.item() == 0]

#     # 确保有足够的正负样本
#     if len(positive_indices) < n_query // 2 or len(negative_indices) < n_query // 2:
#         # 如果正样本或负样本不足，调整 n_query 以适应实际情况
#         max_possible_query = min(len(positive_indices) + len(negative_indices), n_query)
#         n_query = max_possible_query
#         positive_query_count = min(len(positive_indices), n_query // 2)
#         negative_query_count = n_query - positive_query_count

#         positive_query = random.sample(positive_indices, positive_query_count)
#         negative_query = random.sample(negative_indices, negative_query_count)
#     else:
#         positive_query = random.sample(positive_indices, n_query // 2)
#         negative_query = random.sample(negative_indices, n_query // 2)

#     # 合并正负样本查询集
#     query_list = positive_query + negative_query
#     random.shuffle(query_list)

#     # 检查 data 是否是一个 InMemoryDataset 类型的对象。
#     if isinstance(data, InMemoryDataset):
#         support_dataset = data[torch.tensor(support_list)]
#         query_dataset = data[torch.tensor(query_list)]
#     elif isinstance(data, Dataset):
#         support_dataset = Subset(data, support_list)
#         query_dataset = Subset(data, query_list)
    
#     return support_dataset, query_dataset




def sample_datasets(data, dataset, task, n_way, m_support, n_query):
    distri_list = obtain_distr_list(dataset)
    
    support_list = random.sample(range(0, distri_list[task][0]), m_support)
    support_list += random.sample(range(distri_list[task][0],len(data)), m_support)
    random.shuffle(support_list)

    l = [i for i in range(0, len(data)) if i not in support_list]
    query_list = random.sample(l, n_query)

    if isinstance(data, InMemoryDataset):
        support_dataset = data[torch.tensor(support_list)]
        query_dataset = data[torch.tensor(query_list)]
    elif isinstance(data, Dataset):
        support_dataset = Subset(data, support_list)
        query_dataset = Subset(data, query_list)
    
    return support_dataset, query_dataset

def sample_test_datasets(data, dataset, task, n_way, m_support, n_query):
    distri_list = obtain_distr_list(dataset)
    
    support_list = random.sample(range(0, distri_list[task][0]), m_support)
    support_list += random.sample(range(distri_list[task][0],len(data)), m_support)
    random.shuffle(support_list)

    l = [i for i in range(0, len(data)) if i not in support_list]
    random.shuffle(l)

    if isinstance(data, InMemoryDataset):
        support_dataset = data[torch.tensor(support_list)]
        query_dataset = data[torch.tensor(l)]
    elif isinstance(data, Dataset):
        support_dataset = Subset(data, support_list)
        query_dataset = Subset(data, l) 
               
    return support_dataset, query_dataset



