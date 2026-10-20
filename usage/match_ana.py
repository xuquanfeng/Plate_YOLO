import os
import numpy as np
import pandas as pd
import json
from scipy.spatial import cKDTree  # 引入 KDTree 进行坐标匹配

# def run(fits_path, output_dir):
def compute_iou(box1, boxes2):
    """
    计算一个框 (box1) 与 多个框 (boxes2) 的 IoU
    box格式: [xmin, ymin, xmax, ymax]
    """
    # 计算交集 (Intersection) 的左上角和右下角
    # box1: (4,)  boxes2: (N, 4)
    ixmin = np.maximum(box1[0], boxes2[:, 0])
    iymin = np.maximum(box1[1], boxes2[:, 1])
    ixmax = np.minimum(box1[2], boxes2[:, 2])
    iymax = np.minimum(box1[3], boxes2[:, 3])

    # 计算交集宽和高，如果无重叠则为0
    iw = np.maximum(ixmax - ixmin, 0)
    ih = np.maximum(iymax - iymin, 0)
    
    inter_area = iw * ih
    
    # 计算并集 (Union)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    boxes2_area = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    union_area = box1_area + boxes2_area - inter_area
    
    # 防止除以0
    return inter_area / (union_area + 1e-6)
def match_yolo_to_sextractor(sex_path, yolo_path, iou_thresh=0.25):
    """
    匹配 YOLO 结果与 SExtractor 真星数据。
    
    Args:
        sex_path: SExtractor txt 路径 (x, y, xmin, ymin, xmax, ymax, label)
        yolo_path: YOLO txt 路径 (x, y, w, h, conf) - 假设是像素坐标
        match_radius: 匹配允许的最大距离（像素）
    """
    
    # --------------------------
    # 1. 读取数据
    # --------------------------
    try:
        # 读取 SExtractor (假设空格分隔)
        sex_data = np.loadtxt(sex_path)
        # 读取 YOLO (假设空格分隔)
        yolo_data = np.loadtxt(yolo_path)
    except Exception as e:
        print(f"读取文件失败: {e}")
        return

    # 处理 YOLO 为空的情况
    if len(yolo_data) == 0:
        print("YOLO 文件为空，没有检测到目标。")
        return
    if yolo_data.ndim == 1:
        yolo_data = yolo_data.reshape(1, -1)

    # --------------------------
    # 2. 筛选 SExtractor 真星
    # --------------------------
    # sex_data 格式: [x, y, xmin, ymin, xmax, ymax, label]
    # label != 0 代表真实星/星系
    real_star_mask = sex_data[:, 6] != 0
    real_stars = sex_data[real_star_mask]
    
    # # 提取真星坐标 (x, y) -> 第0列和第1列
    # gt_coords = real_stars[:, 0:2]

    # print(f"SExtractor 总源数: {len(sex_data)}")
    # print(f"其中真实星(Label!=0): {len(real_stars)}")
    # print(f"YOLO 检测总数: {len(yolo_data)}")
    # print("-" * 30)

    gt_data = sex_data[sex_data[:, 6] != 0]
    
    # 提取坐标 [xmin, ymin, xmax, ymax] -> 对应列索引 2, 3, 4, 5
    gt_boxes = gt_data[:, 2:6] 
    
    # print(f"真星(GT)数量: {len(gt_boxes)}")
    # print(f"YOLO检测数量: {len(yolo_data)}")

    if len(gt_boxes) == 0:
        print("没有有效的真星数据。")
        return 0
    # --- 3. 准备预测值 (Prediction) ---
    # yolo_data 格式: [x_center, y_center, width, height, conf]
    # 需要转换为 [xmin, ymin, xmax, ymax]
    yolo_boxes = np.zeros((len(yolo_data), 4))
    
    x_c = yolo_data[:, 0]
    y_c = yolo_data[:, 1]
    w   = yolo_data[:, 2]
    h   = yolo_data[:, 3]
    
    yolo_boxes[:, 0] = x_c - w / 2  # xmin
    yolo_boxes[:, 1] = y_c - h / 2  # ymin
    yolo_boxes[:, 2] = x_c + w / 2  # xmax
    yolo_boxes[:, 3] = y_c + h / 2  # ymax
    yolo_confs = yolo_data[:, 4]     # 置信度

    # --- 4. 计算 IoU 矩阵 & 匹配 ---
    # 这里的逻辑是：
    # 1. 对每个 YOLO 框，找出和它 IoU 最大的 GT 框
    # 2. 如果 IoU > 阈值，则认为潜在匹配
    # 3. 必须处理“多个 YOLO 框匹配同一个 GT”的情况，只保留置信度最高或 IoU 最高的那个
    
    # 存储所有可能的匹配 (yolo_idx, gt_idx, iou)
    matches = []
    
    for i, y_box in enumerate(yolo_boxes):
        # 计算当前 YOLO 框与所有 GT 框的 IoU
        ious = compute_iou(y_box, gt_boxes)
        
        # 找到 IoU 最大的那个 GT
        max_iou_idx = np.argmax(ious)
        max_iou = ious[max_iou_idx]
        
        if max_iou >= iou_thresh:
            matches.append([i, max_iou_idx, max_iou])
            
    # 将匹配结果转为数组
    matches = np.array(matches)
    
    if len(matches) == 0:
        print(f"在 IoU > {iou_thresh} 下无匹配。")
        return

    # --- 5. 贪婪匹配 (处理冲突) ---
    # 按照 IoU 从大到小排序，优先匹配重合度最高的
    # 这一步是为了防止一个真星被多个 YOLO 框重复统计
    matches = matches[matches[:, 2].argsort()[::-1]]
    
    final_matches = []
    seen_yolo = set()
    seen_gt = set()
    
    for match in matches:
        y_idx = int(match[0])
        g_idx = int(match[1])
        iou_val = match[2]
        
        if y_idx not in seen_yolo and g_idx not in seen_gt:
            # 这是一个有效的唯一匹配
            final_matches.append((y_idx, g_idx, iou_val))
            seen_yolo.add(y_idx)
            seen_gt.add(g_idx)
            
    # --- 6. 统计结果 ---
    tp = len(final_matches)  # True Positives (匹配上的)
    # fp = len(yolo_data) - tp # False Positives (YOLO有，但没匹配上真星 -> 伪影或虚警)
    # fn = len(gt_boxes) - tp  # False Negatives (真星有，YOLO没检测到 -> 漏检)
    
    # print("-" * 30)
    # print(f"IoU 阈值: {iou_thresh}")
    # print(f"匹配成功 (True Positive): {tp}")
    # print(f"误报/伪影 (False Positive): {fp}")
    # print(f"漏检 (False Negative): {fn}")
    
    # if len(yolo_data) > 0:
    #     print(f"查准率 (Precision): {tp / len(yolo_data):.2%}")
    # if len(gt_boxes) > 0:
    #     print(f"查全率 (Recall): {tp / len(gt_boxes):.2%}")
        
    # # 计算匹配目标的平均置信度
    # if tp > 0:
    #     matched_indices = [m[0] for m in final_matches]
    #     avg_conf = np.mean(yolo_confs[matched_indices])
    #     print(f"匹配目标的平均置信度: {avg_conf:.4f}")
    return len(sex_data), len(real_stars), len(yolo_data), tp

    # # --------------------------
    # # 3. 准备 YOLO 坐标
    # # --------------------------
    # # yolo_data 格式: [x, y, w, h, conf]
    # # 假设这里的 x, y 已经是中心点像素坐标。
    # # 如果 YOLO 输出是归一化的 (0-1)，这里需要乘以图像宽高还原!
    # pred_coords = yolo_data[:, 0:2] 

    # # --------------------------
    # # 4. 执行匹配 (KDTree)
    # # --------------------------
    # if len(gt_coords) == 0:
    #     print("没有真实星用于匹配。")
    #     return

    # # 构建真值的 KDTree
    # tree = cKDTree(gt_coords)

    # # 查询: 为每个 YOLO 点找最近的真星
    # # dists: 距离
    # # indices: 对应真星在 real_stars 中的索引
    # dists, indices = tree.query(pred_coords, k=1)

    # # --------------------------
    # # 5. 统计结果
    # # --------------------------
    # # 判定匹配成功：距离小于阈值
    # matched_mask = dists <= match_radius
    
    # num_matched = np.sum(matched_mask)
    # num_unmatched = len(yolo_data) - num_matched

    # # 获取匹配上的 YOLO 目标的置信度
    # matched_confs = yolo_data[matched_mask, 4]
    # avg_conf = np.mean(matched_confs) if len(matched_confs) > 0 else 0

    # print(f"匹配半径阈值: {match_radius} 像素")
    # print(f"YOLO 中匹配上真星的数量: {num_matched}")
    # print(f"YOLO 中未匹配/匹配伪影的数量: {num_unmatched}")
    
    # if len(yolo_data) > 0:
    #     precision = num_matched / len(yolo_data)
    #     print(f"YOLO 查准率 (Precision): {precision:.2%}")
        
    # if len(real_stars) > 0:
    #     recall = num_matched / len(real_stars) # 注意：这里简单假设一个星对应一个YOLO框
    #     print(f"YOLO 查全率 (Recall): {recall:.2%} (检测到了多少比例的真星)")

    # print(f"匹配目标的平均置信度: {avg_conf:.4f}")

    # (可选) 如果你想保存匹配上的结果
    # matched_yolo_data = yolo_data[matched_mask]
    # np.savetxt("yolo_matched.txt", matched_yolo_data)

# --- 运行示例 ---
# 请修改为你的实际文件路径
# match_yolo_to_sextractor('sex_output.indata', 'yolo_output.txt', match_radius=5.0)

if __name__ == "__main__":
    save_json = 'run2.json'
    if not os.path.exists(save_json):
        with open(save_json, 'w') as f:
            f.write('{}')
    else:
        with open(save_json, 'r') as f:
            content = f.read()
            if content.strip() == '':
                with open(save_json, 'w') as f_w:
                    f_w.write('{}')
    yolo_path = '/home/yaowei/lenovo_1/xqf/Uzbekistan_plate'
    sex_path = '/home/yaowei/lenovo_1/xqf/Uzbekistan_plate_sex'
    with open('/home/yaowei/lenovo_1/xqf/yolov5/run1.json', 'r') as f:
        content = json.load(f)
    # print(content.keys())
    for num, row in enumerate(content.keys()):
        print(f'[{num}/{len(content.keys())}]')
        with open(save_json, 'r') as f:
            data = json.load(f)
        if row in data.keys():
            print('runed! pass!')
            continue
        yolo_path_f = os.path.join(yolo_path, row+'.txt')
        sex_path_f = os.path.join(sex_path, row+'.txt')
        if os.path.exists(yolo_path_f) and os.path.exists(sex_path_f):
            resu = match_yolo_to_sextractor(sex_path_f, yolo_path_f)
            if resu is None:
                print('match_yolo_to_sextractor return None! pass!')
                data[row] = 'match_yolo_to_sextractor return None'
                with open(save_json, 'w') as f:
                    json.dump(data, f, indent=4)
                continue
            a1, a2, a3, a4 = resu
            data[row] = {'sex_num':content[row]['num'], 'first_num':a1, 'real_star_num':a2, 'yolo_num':a3, 'match_num':a4,
                         'yolo_P': (f'{a4/a3:.2%}' if a3>0 else '0'), 'yolo_R': (f'{a4/a2:.2%}' if a2>0 else '0')}
            with open(save_json, 'w') as f:
                json.dump(data, f, indent=4)
        else:
            print('txt_path not exist! pass!')
            data[row] = 'not exist file'
            with open(save_json, 'w') as f:
                json.dump(data, f, indent=4)
    # run()