航拍洪水分割数据集-无人机拍摄洪涝地物分割数据集![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a32fb571f98d4fd4a378e89f76ac3092.png)


大疆M350RTK无人机采集数据，512*512尺寸，共12000余张图像，共800MB数据量![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7cad8cb44fc846a9838c98f265fedd98.png)


# 用于语义分割标签值，包含6类分割标签![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7ae96c52f18d41f5ada22f6407c9cdb8.png)


1 | water | 水体 | 128,0,128
2 | building | 建筑 | 128,128,0
3 | building_flooded | 被淹没建筑 | 0,0,128
4 | road | 道路 | 128,0,0
5 | road_flooded | 被淹没道路 | 0,128,0
6 | background | 背景 | 0,0,0

![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/66030e39b19342469d7cc36780d84b6d.png)
# 无人机拍摄洪涝地物分割数据集

## 一、数据集基础信息

| 项目 | 内容 |
|---|---|
| 数据集名称 | 无人机洪涝地物语义分割数据集 |
| 采集设备 | 大疆 M350 RTK 无人机 |
| 图像尺寸 | 512 × 512 像素 |
| 图像数量 | 12000 余张 |
| 数据量 | 约 800MB |
| 标注方式 | 像素级语义分割标签（PNG 掩码） |
| 任务类型 | 语义分割（Semantic Segmentation） |
| 类别总数 | 6 类 |

## 二、类别标签与颜色映射

| 类别ID | 类别名称 | 中文含义 | RGB颜色值 |
|:---:|---|---|---|
| 1 | water | 水体 | 128, 0, 128 |
| 2 | building | 建筑 | 128, 128, 0 |
| 3 | building_flooded | 被淹没建筑 | 0, 0, 128 |
| 4 | road | 道路 | 128, 0, 0 |
| 5 | road_flooded | 被淹没道路 | 0, 128, 0 |
| 6 | background | 背景 | 0, 0, 0 |

> 说明：标签值为 ID 1-6（对应上表），背景 ID=6。通常训练时将类别ID映射为 0-5（即 water=0, building=1, building_flooded=2, road=3, road_flooded=4, background=5）。

## 三、数据目录结构建议

```text
flood_seg_dataset/
├── images/
│   ├── train/       # 训练图像
│   ├── val/         # 验证图像
│   └── test/        # 测试图像
├── masks/
│   ├── train/       # 训练掩码
│   ├── val/         # 验证掩码
│   └── test/        # 测试掩码
└── dataset.yaml
```

---

## 四、YOLOv11-Seg 训练代码

### 1. dataset.yaml

```yaml
path: ./flood_seg_dataset
train: images/train
val: images/val
test: images/test

names:
  0: water
  1: building
  2: building_flooded
  3: road
  4: road_flooded
  5: background
```

### 2. 训练代码 train.py

```python
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo11m-seg.pt")

    model.train(
        data="dataset.yaml",
        epochs=100,
        imgsz=512,          # 与原图尺寸一致
        batch=16,
        device=0,
        patience=20,
        cos_lr=True,
        mosaic=0.6,
        mixup=0.2,
        hsv_h=0.01,
        hsv_s=0.5,
        hsv_v=0.3,
        fliplr=0.5,
        project="runs/train",
        name="flood_seg"
    )
```

### 3. 验证代码 val.py

```python
from ultralytics import YOLO

model = YOLO("./runs/train/flood_seg/weights/best.pt")

metrics = model.val(
    data="dataset.yaml",
    imgsz=512,
    device=0
)

print(f"mIoU: {metrics.seg.map:.4f}")
print(f"mIoU@0.5: {metrics.seg.map50:.4f}")
print(f"Precision: {metrics.seg.mp:.4f}")
print(f"Recall: {metrics.seg.mr:.4f}")
```

### 4. 推理可视化代码 predict.py

```python
from ultralytics import YOLO
import cv2

model = YOLO("./runs/train/flood_seg/weights/best.pt")

results = model.predict(
    source="./test_images",
    imgsz=512,
    conf=0.25,
    save=True,               # 保存检测可视化图
    save_txt=True,           # 保存检测结果
    project="runs/predict",
    name="flood_seg_result"
)

# 输出每个类别的像素占比（可用于计算淹没面积）
for res in results:
    masks = res.masks
    if masks is None:
        print("未检测到目标")
        continue
    img_area = res.orig_shape[0] * res.orig_shape[1]
    for i, cls_id in enumerate(res.boxes.cls):
        cls_name = model.names[int(cls_id)]
        pixel_area = masks.data[i].sum().item()
        ratio = pixel_area / img_area * 100
        print(f"{cls_name}: {ratio:.2f}%")
```

---

## 五、应用场景

| 应用场景 | 具体用途 |
|---|---|
| 洪涝灾害评估 | 快速识别被淹没的建筑和道路，评估受灾范围 |
| 应急救援 | 洪水发生后确定可通行道路，规划救援路线 |
| 灾后损失统计 | 统计被淹没建筑数量，为理赔和补助提供依据 |
| 城市内涝监测 | 实时监测城市水体漫延情况，辅助预警 |
| 防洪规划 | 分析历史淹没区域，优化防洪设施布局 |
| 遥感GIS制图 | 生成洪水淹没专题图，与GIS系统结合 |


