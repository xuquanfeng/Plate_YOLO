import os
import numpy as np
import pandas as pd
import json
from astropy.io import ascii
from scipy.spatial import cKDTree

def match_yolo_with_custom_cat(yolo, sex_cat_path, output_path, match_radius=10.0):
    """
    针对用户提供的 Specific SExtractor Catalog 进行 YOLO 匹配
    """
    
    # --- 1. 读取 SExtractor 目录 ---
    print(f"正在读取 catalog 文件: {sex_cat_path} ...")
    try:
        # astropy 通常能自动解析 SExtractor 格式的 header
        sex_data = ascii.read(sex_cat_path)
    except Exception as e:
        print(f"读取失败: {e}")
        return None

    # --- 2. 提取数据列 (基于你提供的 Header) ---
    try:
        # 必须存在的列
        sex_x = sex_data['X_IMAGE']
        sex_y = sex_data['Y_IMAGE']
        sex_mag = sex_data['MAG_AUTO']
        
        # 可选列：MAGERR_AUTO (你的 header 里没有，所以这里做个判断，防止报错)
        if 'MAGERR_AUTO' in sex_data.colnames:
            sex_magerr = sex_data['MAGERR_AUTO']
        else:
            # 如果没有误差列，填入 NaN
            sex_magerr = np.full(len(sex_data), np.nan)
            
    except KeyError as e:
        print(f"列名匹配错误: {e}. 请检查 .cat 文件头是否被正确解析。")
        # 调试用：打印读到的所有列名
        print(f"读取到的列名: {sex_data.colnames}")
        return None

    # --- 3. 构建 KDTree (用于高效匹配) ---
    sex_coords = np.column_stack((sex_x, sex_y))
    tree = cKDTree(sex_coords)

    # --- 4. 处理 YOLO 数据并匹配 ---
    matched_results = []
    
    # 统计匹配率用
    match_count = 0
    yolo_boxes = np.loadtxt(yolo, dtype=float)[:,:4]

    for idx, box in enumerate(yolo_boxes):
        xc, yc = box[0], box[1]
        hight, width = box[2], box[3]
        x1 = xc - width / 2
        y1 = yc - hight / 2
        x2 = xc + width / 2
        y2 = yc + hight / 2

        # 在 KDTree 中找最近邻
        dist, sex_idx = tree.query([xc, yc], k=1)

        result_row = {
            'yolo_xc': xc,
            'yolo_yc': yc,
            'yolo_h': hight,
            'yolo_w': hight,
            # 'yolo_x1': x1,
            # 'yolo_y1': y1,
            # 'yolo_x2': x2,
            # 'yolo_y2': y2,
            'sex_mag_auto': np.nan,
            'sex_magerr': np.nan,
            'match_dist': dist
        }

        # 判定距离
        if dist <= match_radius:
            result_row['sex_mag_auto'] = sex_mag[sex_idx]
            result_row['sex_magerr'] = sex_magerr[sex_idx]
            match_count += 1
        
            matched_results.append(result_row)

    # --- 5. 输出统计 ---
    total = len(yolo_boxes)
    # print(f"匹配完成: {match_count}/{total} ({(match_count/total)*100:.1f}%) 的 YOLO 框找到了对应的源。")

    # --- 6. 保存结果 ---
    pd.DataFrame(matched_results).to_csv(output_path, index=False)
    return total, (match_count/total)*100

if __name__ == "__main__":
    save_json = 'run_fail1.json'
    if not os.path.exists(save_json):
        with open(save_json, 'w') as f:
            f.write('{}')
    else:
        with open(save_json, 'r') as f:
            content = f.read()
            if content.strip() == '':
                with open(save_json, 'w') as f_w:
                    f_w.write('{}')
    output_path = '/home/yaowei/lenovo_1/xqf/Uzbekistan_plate_yolo_sex_fail'
    os.makedirs(output_path, exist_ok=True)
    yolo = '/home/yaowei/lenovo_1/xqf/Uzbekistan_plate_fail'
    pdd = pd.read_csv('/home/yaowei/lenovo_1/xqf/yolov5/mag/20260128-单次－有日期底片TNA和DAZ－处理情况全汇总-初匹配失败.csv')
    # pdd = pd.read_csv('/home/yaowei/lenovo_1/xqf/yolov5/20260119-单次－有日期底片－处理情况全汇总-成功.csv')
    for num, row in enumerate(pdd.iloc[:,0]):
        fits_path = row.replace('home/dpk/nas1_shuju1','home/yaowei/nas_31')+'.fits'
        print(f'[{num}/{len(pdd.iloc[:,0])}]',fits_path)
        name = fits_path.split('/')[-1].replace('.fits','')
        yolo_path = f'{yolo}/{name}.txt'
        cat_file_path = fits_path.replace('.fits','.cat')
        output_txt_path = f'{output_path}/{name}.txt'
        with open(save_json, 'r') as f:
            data = json.load(f)
        if name in data.keys():
            print('runed! pass!')
            continue
        if os.path.exists(yolo_path):
            tot, match_rate = match_yolo_with_custom_cat(yolo_path, cat_file_path, output_txt_path, match_radius=10.0)
            # num = run(fits_path, output_path)
            data[name] = {'path':fits_path, 'total_num':tot, 'match_rate':match_rate}
            with open(save_json, 'w') as f:
                json.dump(data, f, indent=4)
        else:
            print(f'{fits_path} not exist! pass!')
            data[name] = 'not exist'
            with open(save_json, 'w') as f:
                json.dump(data, f, indent=4)