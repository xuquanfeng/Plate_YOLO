import os
import shutil
import numpy as np
from astropy.io import fits
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import glob
import torch
import pandas as pd
import json
from torchvision.ops import nms
# 导入你的YOLOv5推理函数
# 假设你的推理代码文件名为 detect.py
try:
    from detect import run as yolo_run
except ImportError:
    print("错误: 无法导入 detect.py。请确保此脚本与 detect.py 在同一目录下。")
    exit()

def normalize_to_uint8(data):
    """
    将 FITS 的任意数值范围归一化到 0-255 (uint8)
    """
    # 处理 NaN (空值)，将其变为最小值
    data = np.nan_to_num(data)
    
    min_val = np.min(data)
    max_val = np.max(data)
    
    # 防止除以 0 (如果图像全是黑的或全是同一个值)
    if max_val == min_val:
        return np.zeros(data.shape, dtype=np.uint8)
    
    # 线性拉伸公式: (x - min) / (max - min) * 255
    norm_data = (data - min_val) / (max_val - min_val) * 255.0
    
    return norm_data.astype(np.uint8)

def xywh2xyxy(x):
    # 将 center_x, center_y, w, h 转换为 x1, y1, x2, y2
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def merge_yolo_labels(
    labels_dir, 
    output_path="merged_result.txt", 
    tile_size=(640, 640),
    conf_thres=0.0,
    iou_thres=0.1
):
    """
    labels_dir: YOLO推理结果保存的 labels 文件夹路径 (例如 runs/detect/exp/labels)
    original_fits_path: 原始 FITS 大图路径 (用于获取大图尺寸进行归一化)
    tile_size: 切割时的小图尺寸 (width, height)
    """

    txt_files = glob.glob(os.path.join(labels_dir, "*.txt"))
    if not txt_files:
        print("未找到任何标签文件！")
        return

    all_boxes = []   # 格式: [x1, y1, x2, y2, conf, class_id] (绝对像素坐标)

    print(f"开始合并 {len(txt_files)} 个切片标签...")

    # 2. 遍历所有小图标签
    for txt_file in txt_files:
        # 解析文件名 tile_{x}_{y}.txt
        filename = os.path.basename(txt_file)
        try:
            parts = filename.replace('.txt', '').split('_')
            # 假设文件名格式严格为 tile_x_y (例如 tile_0_640)
            off_x = int(parts[-2])
            off_y = int(parts[-1])
        except:
            print(f"跳过文件名格式错误的文件: {filename}")
            continue

        # 读取内容
        with open(txt_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            data = list(map(float, line.strip().split()))
            if len(data) < 6: continue # 确保有 conf (yolov5 save_conf=True)
            
            cls_id = int(data[0])
            cx_rel, cy_rel, w_rel, h_rel = data[1:5]
            conf = data[5]

            # 过滤低置信度 (虽然推理时已过滤，这里可二次过滤)
            if conf < conf_thres:
                continue

            # A. 还原为小图的一般像素坐标 (Pixel in Tile)
            w_pixel = w_rel * tile_size[0]
            h_pixel = h_rel * tile_size[1]
            cx_pixel = cx_rel * tile_size[0]
            cy_pixel = cy_rel * tile_size[1]

            # B. 加上偏移量，转为大图绝对坐标 (Pixel in Global)
            cx_global = cx_pixel + off_x
            cy_global = cy_pixel + off_y
            
            # 转为 x1, y1, x2, y2 用于 NMS
            x1 = cx_global - w_pixel / 2
            y1 = cy_global - h_pixel / 2
            x2 = cx_global + w_pixel / 2
            y2 = cy_global + h_pixel / 2

            all_boxes.append([x1, y1, x2, y2, conf, cls_id])

    if not all_boxes:
        print("没有检测到任何目标。")
        return

    # 3. 全局 NMS (去重)
    boxes_tensor = torch.tensor(all_boxes)
    
    # 提取坐标和置信度
    coords = boxes_tensor[:, :4]
    scores = boxes_tensor[:, 4]
    
    # 执行 NMS
    # 注意：如果是多类别，通常需要按类别分别做 NMS。
    # 这里简单起见做了一个不分通过类别的 NMS，如果你需要分科，可以循环类别处理。
    keep_indices = nms(coords, scores, iou_thres)
    final_boxes = boxes_tensor[keep_indices].numpy()

    print(f"合并前检测框数: {len(all_boxes)}")
    print(f"NMS去重后检测框数: {len(final_boxes)}")
    with open(output_path, 'w') as f_txt:
        for box in final_boxes:
            x1, y1, x2, y2, conf, cls = box
            # if cls == 0:
            #     continue
            
            # 计算大图下的相对坐标
            w = (x2 - x1)
            h = (y2 - y1)
            cx = x1 + w / 2
            cy = y1 + h / 2
            
            f_txt.write(f"{cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {conf:.6f} {int(cls)}\n")
    # csv_path = output_path.replace('.txt', '.csv')
    # with open(csv_path, 'w') as f_csv:
    #     f_csv.write("class_id,x_global,y_global,w_global,h_global,confidence\n")
    #     for box in final_boxes:
    #         x1, y1, x2, y2, conf, cls = box
    #         w = x2 - x1
    #         h = y2 - y1
    #         cx = x1 + w / 2
    #         cy = y1 + h / 2
    #         f_csv.write(f"{int(cls)},{cx:.2f},{cy:.2f},{w:.2f},{h:.2f},{conf:.4f}\n")
    
    # print(f"结果已保存至:\n  1. {csv_path} (绝对像素坐标)\n  2. {output_path} (大图YOLO相对坐标)")
    # 格式 2: YOLO 归一化格式 (相对于大图的 0-1)


def slice_fits_to_png(fits_path, output_dir, tile_size=(640, 640), overlap=0):
    """
    读取FITS大图，切割并保存为PNG。
    文件名格式: tile_{x}_{y}.png (x, y 为在大图中的左上角坐标)
    """
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)  # 清空旧数据
    os.makedirs(output_dir)

    print(f"正在读取 FITS 文件: {fits_path} ...")
    with fits.open(fits_path) as hdul:
        # 假设图像数据在 Primary HDU (ext=0)
        original_data = hdul[0].data
        
        # 假如是多维数据 (例如 [1, H, W])，取最后两维
        if original_data.ndim == 3:
            original_data = original_data[0]
            
        h, w = original_data.shape
        print(f"原始图像尺寸: {w}x{h}")

        # # 归一化转为 uint8
        # print("正在归一化图像数据...")
        # img_uint8 = normalize_fits_data(original_data)

    step_x = tile_size[0] - overlap
    step_y = tile_size[1] - overlap
    
    # 生成切片坐标
    tiles = []
    for y in range(0, h, step_y):
        for x in range(0, w, step_x):
            # 简单的边界处理：如果最后一块不够大，就让它“倒退”一点以保证尺寸一致
            # 或者直接切下来，YOLO 的 LoadImages 会处理尺寸
            
            # 这里我们选择直接切，并处理边缘填充（如果需要）
            # 但最简单且对YOLO最友好的是：如果越界，就取图像最后 tile_size 的大小
            cur_x = x
            cur_y = y
            
            # 如果当前块超出了边界
            if cur_x + tile_size[0] > w:
                cur_x = w - tile_size[0]
            if cur_y + tile_size[1] > h:
                cur_y = h - tile_size[1]
                
            # 防止图片本身比 tile_size 还小的情况
            cur_x = max(0, cur_x)
            cur_y = max(0, cur_y)
            
            tiles.append((cur_x, cur_y))

    # 去重（因为上面的边界处理可能导致重复添加最后的块）
    tiles = sorted(list(set(tiles)))

    print(f"开始切割，共 {len(tiles)} 张子图...")
    
    for (x, y) in tqdm(tiles):
        # 切割
        crop = original_data[y : y + tile_size[1], x : x + tile_size[0]]
        
        # 保存为 PNG
        # 文件名记录坐标: tile_x坐标_y坐标.png
        save_name = os.path.join(output_dir, f"tile_{x}_{y}.png")
        img_uint8 = normalize_to_uint8(crop)
        img = Image.fromarray(img_uint8, mode='L') # 'L' 代表 8-bit pixels, black and white
        img.save(save_name)

    return output_dir

def run(large_fits_path, output_txt_path):
    # ================= 配置区域 =================
    # large_fits_path = '/data/public/xqf/temp/BJ8503SD2623001.fits'  # 修改为你的 FITS 文件路径
    weights_path = '/home/yaowei/lenovo_1/xqf/yolov5/runs/train/exp/weights/best.pt'                # 修改为你的权重路径
    temp_png_dir = 'temp_inference_tiles'      # 临时存放 PNG 的文件夹
    project_save_dir = 'runs/detect_large_fits' # 结果保存路径
    
    tile_w, tile_h = 256, 256                  # 切割大小 (建议与模型输入大小一致)
    overlap = 10                              # 重叠区域 (防止目标在切分线上被切断)
    # ===========================================

    # 1. 切割 FITS 并转为 PNG
    # 这一步生成的文件会自动适配 LoadImages 的读取逻辑
    slice_fits_to_png(large_fits_path, temp_png_dir, tile_size=(tile_w, tile_h), overlap=overlap)

    # 2. 调用 YOLOv5 推理
    # 直接调用你提供的 detect.py 中的 run 函数
    # source 设置为刚刚生成的 PNG 文件夹路径
    print(f"\n开始 YOLOv5 推理，源目录: {temp_png_dir}")
    
    LABELS_DIR = f"{project_save_dir}/exp/labels"  # 上一步推理生成的 labels 路径
    if os.path.exists(LABELS_DIR):
        shutil.rmtree(LABELS_DIR)  # 清空旧数据
    os.makedirs(LABELS_DIR)
    yolo_run(
        weights=weights_path,
        source=temp_png_dir,  # 指向切好的 PNG 文件夹
        imgsz=(tile_w, tile_h),
        conf_thres=0.25,
        project=project_save_dir,
        name='exp',
        exist_ok=True,
        save_txt=True,    # 必须为 True，否则不生成 txt
        save_conf=True,   # 建议为 True，后续合并大图去重时需要置信度
        nosave=True,      # 【核心设置】设置为 True，表示“不保存图片/视频”
        save_crop=False,  # 确保不保存截取的小图
        view_img=False    # 确保不弹出显示窗口
    )
    
    merge_yolo_labels(
        labels_dir=LABELS_DIR,
        output_path=os.path.join(output_txt_path, large_fits_path.split('/')[-1].replace('.fits', '.txt')),
        # output_path=large_fits_path.split('/')[-1].replace('.fits', '.txt'),
        tile_size=(tile_w, tile_h), # 必须和推理时的 imgsz 以及 切割时的 tile_size 一致
        iou_thres=0.3         # NMS 阈值，越小去除重叠越严格
    )

if __name__ == "__main__":
    save_json = 'run.json'
    if not os.path.exists(save_json):
        with open(save_json, 'w') as f:
            f.write('{}')
    else:
        with open(save_json, 'r') as f:
            content = f.read()
            if content.strip() == '':
                with open(save_json, 'w') as f_w:
                    f_w.write('{}')
    output_path = '/home/yaowei/lenovo_1/xqf/Uzbekistan_plate'
    os.makedirs(output_path, exist_ok=True)
    pdd = pd.read_csv('/home/yaowei/lenovo_1/xqf/yolov5/20260119-单次－有日期底片－处理情况全汇总-成功.csv')
    for num, row in enumerate(pdd.iloc[:,0]):
        fits_path = row.replace('home/dpk/nas1_shuju1','home/yaowei/nas_31')+'.fits'
        print(f'[{num}/{len(pdd.iloc[:,0])}]',fits_path)
        name = fits_path.split('/')[-1].replace('.fits','')
        # output_txt_path = f'{output_path}/{name}.txt'
        with open(save_json, 'r') as f:
            data = json.load(f)
        if name in data.keys():
            print('runed! pass!')
            continue
        if os.path.exists(fits_path):
            run(fits_path, output_path)
            data[name] = fits_path
            with open(save_json, 'w') as f:
                json.dump(data, f, indent=4)
        else:
            print(f'{fits_path} not exist! pass!')
            data[name] = 'not exist'
            with open(save_json, 'w') as f:
                json.dump(data, f, indent=4)
    # run()