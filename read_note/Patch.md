# Patch_attack 类讲解


## 库调用

这是一个使用PyTorch框架实现的`Attacker`类。我将逐行为您解释文件中的代码。从上面的输出中，我们可以看到文件的开始部分。首先，让我们看一下导入的模块和包。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import mmcv
```
这些是基础的PyTorch库导入，用于神经网络的定义、操作和优化，以及数据加载。

- `torch`: PyTorch的主要库。
- `torch.nn` 和 `torch.nn.functional`: 用于定义神经网络层和功能。
- `torch.optim`: 用于定义优化算法，如SGD、Adam等。
- `DataLoader`: 用于批量加载数据。
- `mmcv`: 是一个计算机视觉工具库。

接下来的导入部分：

```python
from copy import deepcopy
import numpy as np
```
- `deepcopy`: 用于创建对象的深拷贝。
- `numpy`: 是Python的数值计算库。

之后是一些特定的导入：
```python
from .base import BaseAttacker
from .builder import ATTACKER
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.models.builder import LOSSES
from mmcv.runner.optimizer import OPTIMIZERS
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
```
这些导入来自于`mmdet`和`mmdet3d`库，它们是用于目标检测和3D目标检测的开源库。


## 装饰该类作为Attacker的子类

最后，有一个装饰器定义：
```python
@ATTACKER.register_module()
```

这个装饰器可能是用来注册这个类作为一个可用的攻击模块。


## 类定义

接下来，我将继续查看代码中的`Attacker`类定义部分。

好的，现在我们看到了`PatchAttack`类的定义，它继承自`BaseAttacker`类。我将为您详细解释这部分代码：

```python
class PatchAttack(BaseAttacker):
```
这定义了一个名为`PatchAttack`的类，该类继承自`BaseAttacker`。

接下来的部分是构造函数的定义，该函数初始化`PatchAttack`类的对象。

```python
def __init__(self,
             step_size,
             num_steps,
             loss_fn,
             assigner,
             patch_size=None,
             dynamic_patch_size=False,
             mono_model=False,
             scale=0.5,
             *args, 
             **kwargs):
```
构造函数接受以下参数：

- `step_size`: 在一次迭代中更新像素值的步长。
- `num_steps`: 生成对抗性补丁的迭代次数。
- `loss_fn`: 对抗性目标函数。
- `assigner`: 分配预测到真实值的类。
- `patch_size`: 对抗性补丁的大小。如果使用动态补丁大小，则此值为None。
- `dynamic_patch_size`: 当激活时，根据对象大小调整补丁大小。
- `mono_model`: 似乎是一个布尔值，但目前的上下文中并未明确其用途,是否为单目模型。
- `scale`: 对象大小的补丁大小比例，取值范围在(0,1)之间。

函数体中首先调用了父类`BaseAttacker`的构造函数：

```python
super().__init__(*args, **kwargs)
```

接着，构造函数将参数值赋给类的实例变量，并进行一些初始化工作：

```python
self.step_size = step_size
self.num_steps = num_steps
self.dynamic_patch = dynamic_patch_size
self.mono_model = mono_model
self.scale = scale
self.assigner = BBOX_ASSIGNERS.build(assigner)
self.loss_fn = LOSSES.build(loss_fn)
```

接下来，根据`patch_size`是否为None，给`self.patch_size`赋值：

```python
if patch_size is not None:
    self.patch_size = torch.tensor(patch_size)
```

最后，这段代码有几个断言，确保只激活一个补丁大小或动态补丁大小，并确保比例`scale`在(0,1)范围内。

接下来，我将继续查看`PatchAttack`类中的其他方法。

接下来的代码片段展示了`PatchAttack`类中的另一个方法。让我们继续分析：

首先，代码处理了`mono_model`和`step_size`的情况：
```python
if mono_model:
    self.size = (1, 3, 1, 1)  # do not have stereo camera information
else:
    self.size = (1, 1, 3, 1, 1)
```
如果`mono_model`为真（意味着模型是单目的，没有立体摄像机信息），则`size`设置为`(1, 3, 1, 1)`。否则，它被设置为`(1, 1, 3, 1, 1)`。

接着，代码处理了`step_size`是否为列表或元组的情况：
```python
if isinstance(step_size, list or tuple):
    self.step_size = torch.tensor(step_size).view(self.size)
```
如果`step_size`是列表或元组，它会被转换为一个PyTorch张量并被reshape为`self.size`。



## Run函数

然后，定义了`run`方法：
```python
def run(self, model, img, img_metas, gt_bboxes_3d, gt_labels_3d):
```
这个方法似乎是进行补丁攻击优化的核心方法。

方法的参数：
- `model`: 要被攻击的模型。
- `img`: 图像数据，形状为[B, M, C, H, W]。
- `img_metas`: 图像的元数据。
- `gt_bboxes_3d`: 3D边界框的真实值。
- `gt_labels_3d`: 标签的真实值。

接下来的代码进行了一些预处理操作：
- 将模型设置为评估模式。
- 深度复制图像并获取其第一个批次的数据。
- 检查批次大小是否为1。
- 获取图像的通道、高度和宽度。
- 获取3D边界框的真实值。

接着，代码检查3D边界框的真实值是否为空：
```python
if len(gt_bboxes_3d_.tensor) == 0:
    return {'img': img, 'img_metas':img_metas}
```
如果为空，则直接返回图像和其元数据。

然后，代码复制了3D边界框的重心和角点。

如果`mono_model`为真，

首先，如果是单目模型，它会进行从摄像机坐标到激光雷达坐标的转换：
```python
center, corners = self.camera2lidar(center, corners, img_metas)
```

**`camera2lidar`方法**：

```python

def camera2lidar(self, center, corners, img_metas):
    """Convert camera coordinate to lidar coordinate
    """
    assert 'sensor2lidar_translation' in list(img_metas[0].data[0][0].keys())
    assert 'sensor2lidar_rotation' in list(img_metas[0].data[0][0].keys())

    sensor2lidar_translation = np.array(img_metas[0].data[0][0]['sensor2lidar_translation'])
    sensor2lidar_rotation = np.array(img_metas[0].data[0][0]['sensor2lidar_rotation'])

    center = center @ sensor2lidar_rotation.T + sensor2lidar_translation
    corners = corners @ sensor2lidar_rotation.T + sensor2lidar_translation

    return center, corners

```
**这个方法将摄像机坐标转换为激光雷达坐标**。其参数为：
- `center`: 中心点坐标。
- `corners`: 角点坐标。
- `img_metas`: 图像的元数据。

方法首先检查元数据中是否有必要的键，然后从元数据中提取旋转和平移矩阵。使用这些矩阵，它将中心和角点从摄像机坐标转换为激光雷达坐标，并返回转换后的中心和角点。\


接着，代码对中心进行了处理，增加了一个维度来表示它在齐次坐标下的值（也就是末尾添加了一个1）：

```python
center = torch.cat(
    (center, torch.ones_like(center[..., :1])), -1).unsqueeze(dim=-1)
```
> 所谓齐次坐标就是将一个原本是n维的向量用一个n+1维向量来表示。 在空间直角坐标系中，任意一点可用一个三维坐标矩阵[x y z]表示。 如果将该点用一个四维坐标的矩阵[Hx Hy Hz H]表示时，则称为齐次坐标表示方法。 在齐次坐标中，最后一维坐标H称为比例因子。

然后，从`img_metas`中提取了`lidar2img`矩阵，并进行了一些转换：

```python
lidar2img = img_metas[0].data[0][0]['lidar2img']
lidar2img = np.asarray(lidar2img)
lidar2img = center.new_tensor(lidar2img).view(-1, 1, 4, 4)  # (M, 1, 4, 4)
```

使用这个矩阵，它将激光雷达坐标转换为摄像机坐标：

```python
reference_points_cam = torch.matmul(lidar2img.to(torch.float32),
                                    center.to(torch.float32)).squeeze(-1)
```

接着，代码将3D点投影到图像平面上，并对无效的投影进行了过滤：

```python
# filter out invalid project: object center can be seen only by subset of camera
eps = 1e-5
bev_mask = (reference_points_cam[..., 2:3] > eps)
reference_points_cam[..., 0] /= W
reference_points_cam[..., 1] /= H
bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
            & (reference_points_cam[..., 1:2] < 1.0)
            & (reference_points_cam[..., 0:1] < 1.0)
            & (reference_points_cam[..., 0:1] > 0.0))
```
- 详细讲解：

    这段代码主要用于处理变量`reference_points_cam`，该变量似乎包含在摄像机坐标系中的参考点，以及为这些参考点创建一个与之对应的二进制掩码`bev_mask`。以下是对代码的逐步解释：

    1. **定义一个小值**：
    ```python
    eps = 1e-5
    ```
    这里定义了一个非常小的正值`eps`。这通常用于避免除数为零或进行近似相等性检查。

    2. **创建BEV掩码**：
    ```python
    bev_mask = (reference_points_cam[..., 2:3] > eps)
    ```
    这一行创建了一个二进制掩码，该掩码标记了哪些参考点在摄像机坐标系的z轴上有正值（即远离摄像机的点）。这可能用于过滤掉那些在摄像机后面或与摄像机在同一位置的点。

    3. **标准化参考点**：
    ```python
    reference_points_cam[..., 0] /= W
    reference_points_cam[..., 1] /= H
    ```
    这里，`reference_points_cam`中的x坐标（第0维）被除以`W`，而y坐标（第1维）被除以`H`。这将参考点的x和y坐标标准化到[0, 1]范围内，其中`W`和`H`可能分别代表图像的宽度和高度。

    4. **更新BEV掩码**：
    ```python
    bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
                & (reference_points_cam[..., 1:2] < 1.0)
                & (reference_points_cam[..., 0:1] < 1.0)
                & (reference_points_cam[..., 0:1] > 0.0))
    ```
    这里，`bev_mask`被进一步更新以确保标准化后的参考点坐标在[0, 1]范围内。这确保了参考点位于图像的有效范围内。

    总之，这段代码的目的是过滤和标准化摄像机坐标系中的参考点，并为这些点创建一个有效的二进制掩码，该掩码指示哪些点在**图像的有效范围内并且位于摄像机的前面**。

之后，代码计算了图像平面上的3D框有效位置：
```python
reference_points_cam = (reference_points_cam * bev_mask * torch.tensor((W, H))).int()
```

接下来，代码创建了一个与图像形状相同的补丁掩码：
```python
patch_mask = torch.zeros_like(img_)
```

然后，根据是否使用动态补丁，它获取了原始图像的补丁掩码：
```python
if self.dynamic_patch:
    patch_size = self.get_patch_size(corners, lidar2img, bev_mask, scale=self.scale)
patch_mask = self.get_patch_mask(reference_points_cam, bev_mask, patch_mask, \
                                        patch_size if self.dynamic_patch else self.patch_size)
```
- `get_patch_size()`：
  
  1. **方法定义与参数**：
   
    ```python
    def get_patch_size(self, corners, lidar2img, bev_mask, scale=0.5):
    ```

    参数：
    - `corners`：3D边界框的世界坐标角点，形状为[N, 8, 3]。
    - `lidar2img`：世界坐标到图像坐标的转换矩阵，形状为[M, 1, 4, 4]。
    - `bev_mask`：布尔掩码，如果真实中心`n`命中摄像机`m`，则为True。
    - `scale`：补丁大小与图像边界框大小的系数，默认为0.5。

  2. **提取并检查维度**：
   
    ```python
    N, P = corners.size()[:2]
    M = lidar2img.size(0)
    assert P == 8, f"bbox corners should have 8 points, but now {P}"
    ```

    检查边界框角点数量是否为8。

  3. **投影角点到图像平面**：
   
    首先，将角点从[N, 8, 3]转换为[N, 8, 4, 1]，并再次重塑为[8*N, 4, 1]。然后，使用`lidar2img`矩阵将这些角点投影到图像平面上。

  4. **标准化到图像坐标**：

    将投影的角点坐标标准化到图像坐标系。

  5. **计算补丁大小**：
   
    ```python
    xmax = img_corners[..., 0].max(dim=-1)[0]
    xmin = img_corners[..., 0].min(dim=-1)[0]
    ymax = img_corners[..., 1].max(dim=-1)[0]
    ymin = img_corners[..., 1].min(dim=-1)[0]
    ```
    计算每个对象的图像边界框的最大和最小x、y值。然后，使用这些值计算补丁的大小：
    ```python
    patch_size = torch.zeros((M, N, 2))
    patch_size[..., 0] = (scale * (xmax - xmin)).int() 
    patch_size[..., 1] = (scale * (ymax - ymin)).int()
    ```

  6. **返回补丁大小**：
    ```python
    return patch_size
    ```

  总的来说，`get_patch_size`方法根据3D边界框在图像平面上的投影大小计算对抗性补丁的大小。接下来，我将继续查看`PatchAttack`类中的其他代码部分。


- `get_patch_mask()`:

    接下来是`get_patch_mask`方法的定义，它似乎用于计算放置对抗性补丁的位置。此方法接受以下参数：

    - `reference_points_cam`: 在摄像机坐标中的参考点，大小为[M, N, 2]，其中M是摄像机数量，N是真实值数量，2表示(x, y)位置。
    - `bev_mask`: 大小为[M, N, 1]的张量，如果真实值`n`的中心命中摄像机`m`，则为True，否则为False。
    - `patch_mask`: 初始掩码，其中所有位置都为0。
    - `patch_size`: 补丁的大小，默认为(5,5)。

    以下是主要步骤和解释：

    1. **获取维度信息**：
   
    ```python
    if self.mono_model:
        B, C, H, W = patch_mask.size()
        M = 1
    else:
        B, M, C, H, W = patch_mask.size()
    ```
    基于`mono_model`标志，获取补丁掩码的维度信息。

    2. **检查维度一致性**：
    ```python
    M_, N = reference_points_cam.size()[:2]
    ```
    确保摄像机的数量与参考点的数量一致。

    3. **处理补丁大小**：
    
    如果不使用动态补丁，则将补丁大小扩展到与参考点相同的大小。然后，计算每侧的补丁大小。

    ```python
    if not self.dynamic_patch:
        patch_size = patch_size.expand_as(reference_points_cam)

    patch_size = torch.div(patch_size, 2, rounding_mode='floor')
    ```

    4. **计算补丁的边界**：
    计算补丁在图像上的x和y边界。

    ```python
    neg_x = (torch.maximum(reference_points_cam[..., 0] - patch_size[..., 0], torch.zeros_like(reference_points_cam[..., 0])) * bev_mask).int()
    pos_x = (torch.minimum(reference_points_cam[..., 0] + patch_size[..., 0] + 1, W * torch.ones_like(reference_points_cam[..., 0])) * bev_mask).int()
    neg_y = (torch.maximum(reference_points_cam[..., 1] - patch_size[..., 1], torch.zeros_like(reference_points_cam[..., 1])) * bev_mask).int()
    pos_y = (torch.minimum(reference_points_cam[..., 1] + patch_size[..., 1] + 1, H * torch.ones_like(reference_points_cam[..., 1])) * bev_mask).int()
    ```

    这些操作确定补丁在图像上的位置。基本上，它计算参考点周围的区域，以确定应该放置补丁的位置。

    5. **为每个摄像机和参考点设置补丁掩码**：
    ```python
    for m in range(M):
        for n in range(N):
            if neg_x[m, n] == pos_x[m, n]:
                continue
            if self.mono_model:
                patch_mask[0, :, neg_y[m, n] : pos_y[m, n], neg_x[m, n] : pos_x[m, n]] = 1
            else:
                patch_mask[0, m, :, neg_y[m, n] : pos_y[m, n], neg_x[m, n] : pos_x[m, n]] = 1
    ```
    这个双重循环遍历所有摄像机和参考点，并设置相应的补丁掩码。如果`mono_model`为真，则掩码设置为整个图像；否则，它针对每个摄像机设置掩码。

    6. **返回补丁掩码**：
    ```python
    return patch_mask
    ```
    该方法最后返回计算出的补丁掩码。


最后，代码使用像素均值随机初始化补丁，并添加了一个可视化目的的注释行：

```python
x_adv = torch.tensor(self.img_norm['mean']).view(self.size) * torch.randn(img_.shape).to(img_.device).detach()
# x_adv = torch.ones_like(img_) * 255. # this line is for visualization purpose
```


**生成对抗性图像**：

```python
x_adv = x_adv * patch_mask + img_ * (1 - patch_mask)
```
使用补丁掩码`patch_mask`将对抗性补丁`x_adv`与原始图像`img_`融合。

**限制像素范围**：

```python
x_adv = torch.clamp(x_adv, self.lower.view(self.size), self.upper.view(self.size))
```
将`x_adv`的像素值限制在`self.lower`和`self.upper`之间。

**优化**：

在这个循环中，代码对`x_adv`进行优化，使其更加对抗。

```python
for k in range(self.num_steps):
    ...
```
以下是循环中的关键步骤：
   - 使用模型进行预测。
   - 分配预测的边界框到真实值。
   - 如果没有预测分配到真实值，停止攻击。
   - 计算对抗性损失。
   - 使用梯度上升方法更新`x_adv`。
   - 将`x_adv`的像素值再次限制在`self.lower`和`self.upper`之间。

**返回对抗性输入**：

```python
return {'img': img, 'img_metas':img_metas}
```
方法返回修改后的图像和其元数据。



