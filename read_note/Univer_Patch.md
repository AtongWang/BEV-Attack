

# 类：`UniversalPatchAttack`**


## 类注册

```python
@ATTACKER.register_module()
class UniversalPatchAttack(BaseAttacker):
```
这是一个新类，继承自`BaseAttacker`。从其名称来看，它似乎是用于通用补丁攻击的，其中一个固定的补丁模式对所有场景和对象都是对抗的。


## 类定义

好的，我会为每个函数提供逐行的详细解释。由于代码较长，我会从`__init__`方法开始：

### `__init__` 方法:

这是类的初始化方法，用于设置类的初始状态。

```python
def __init__(self,
             epoch,
             dataset_cfg,
             loss_fn,
             assigner,
             lr=100,
             category_specify=True,
             max_train_samples=None,
             is_train=True,
             mono_model=False,
             adv_mode=False,
             catagory_num=10,
             patch_size=50,
             dynamic_patch_size=False,
             scale=0.5,
             patch_path=None,
             *args, 
             **kwargs):
```

- **参数**:
  - `epoch` 到 `patch_path`: 这些参数为类提供了初始设置和配置。
  - `*args, **kwargs`: 这些是可变参数，允许传递任意数量的位置和关键字参数。

```python
    super().__init__(*args, **kwargs)
```
- 调用父类`BaseAttacker`的初始化方法。

```python
    self.epoch = epoch
    self.category_specify = category_specify
    ...
    self.patch_size = torch.tensor(patch_size)
```
- 这些行将方法的参数赋值给类的属性。

```python
    if mono_model:
        self.size = (1, 3, 1, 1) # do not have stereo camera information
    else:
        self.size = (1, 1, 3, 1, 1)
```
- 根据是否是单目模型设置`size`属性。

```python
    assert scale > 0 and scale < 1, f"Scale should be chosen from (0, 1), but now: {scale}"
```
- 确保`scale`参数在(0,1)范围内。

```python
    self.patches = self._init_patch(patch_path)
```
- 调用`_init_patch`方法初始化补丁。

```python
    if is_train:
        self.optimizer = optim.Adam([self.patches], lr=lr)
```
- 如果处于训练模式，则为补丁初始化一个Adam优化器。

## 载入数据加载器 Loader

### `_build_load` 方法:

这个方法从给定的数据集配置中构建数据加载器。

```python
def _build_load(self, dataset_cfg):
    dataset = build_dataset(dataset_cfg.dataset)
```
- 使用`build_dataset`函数从`dataset_cfg`中构建数据集。

```python
    data_loader = build_dataloader(dataset,
                                   samples_per_gpu=1,
                                   workers_per_gpu=dataset_cfg.workers_per_gpu,
                                   dist=False,
                                   shuffle=True)
```
- 使用`build_dataloader`函数为上面创建的数据集构建数据加载器。

```python
    return data_loader
```
- 返回创建的数据加载器。

## 补丁初始化

### `_init_patch` 方法:

这个方法的目的是初始化补丁模式。如果提供了补丁路径，则从该路径加载补丁，否则随机初始化补丁。

```python
def _init_patch(self, patch_paths):
```
- 方法接受一个参数`patch_paths`，它是一个路径列表，指向预训练的补丁文件。

```python
    if patch_paths is not None:
        patches = 0
```
- 检查是否提供了`patch_paths`。如果提供了，初始化`patches`为0。

```python
        for patch_path in patch_paths:
            print(f'Load patch from file {patch_path}')
            info = mmcv.load(patch_path)
```
- 对于`patch_paths`中的每个路径，加载补丁文件。

```python
            patches_ = info['patch'].detach()
```
- 从加载的信息中获取并分离补丁。

```python
            if info['img_norm_cfg']['to_rgb']:
                # a workaround to turn gbr ==> rgb
                patches_ = torch.tensor(patches_.numpy()[:, ::-1].copy())
```
- 如果图像是在GBR格式下标准化的，将其转换为RGB格式。

```python
            patches += patches_
```
- 将当前补丁累加到`patches`中。

```python
        patches /= len(patch_paths)
```
- 将`patches`除以路径的数量，以获取平均值。

```python
        if self.totensor:
            patches /= 255.0
```
- 如果`totensor`属性为True，将`patches`归一化到[0,1]范围。

```python
        patches = (patches - torch.tensor(self.img_norm['mean']).view(1, 3, 1, 1)) / torch.tensor(self.img_norm['std']).view(1, 3, 1, 1)
```
- 使用给定的均值和标准差对`patches`进行标准化。

```python
        return patches
```
- 返回处理后的`patches`。

```python
    catagory_num = self.catagory_num if self.category_specify else 1
```
- 根据`category_specify`属性设置类别数量。

```python
    patches = 255 * torch.rand((catagory_num, 3, self.patch_size[0], self.patch_size[1]))
```
- 如果没有提供`patch_paths`，则随机初始化补丁。

```python
    patches = (patches - torch.tensor(self.img_norm['mean']).view(1, 3, 1, 1)) / torch.tensor(self.img_norm['std']).view(1, 3, 1, 1)
```
- 使用给定的均值和标准差对随机初始化的`patches`进行标准化。

```python
    return patches
```
- 返回处理后的`patches`。


## 补丁训练


### `train` 方法:

这个方法的目的是训练补丁，使其成为一个对目标模型的通用对抗补丁。

```python
def train(self, model):
```
- 方法接受一个参数`model`，这是要攻击的受害者模型。

```python
    model.eval()
```
- 将模型设置为评估模式。这意味着模型中的所有dropout和batchnorm层都将被固定。

```python
    for i in range(self.epoch):
```
- 开始一个循环，持续`epoch`次数，其中`epoch`是类的属性。

```python
        for batch_id, data in enumerate(self.loader):
```
- 对于数据加载器中的每个数据批次，获取批次ID和数据。

```python
            if self.max_train_samples:
                if batch_id + i * len(self.loader) > self.max_train_samples:
                    return self.patches
```
- 如果设置了`max_train_samples`，并且当前的总迭代次数超过了这个值，那么提前返回补丁。

```python
            self._adjust_learning_rate(i * len(self.loader) + batch_id)
```
- 调整学习率。这是一个自定义方法，它可能会根据当前的迭代次数或其他因素来调整学习率。

```python
            self.patches.requires_grad_()
```
- 设置`patches`属性需要梯度，这样在反向传播时可以更新它。

```python
            self.optimizer.zero_grad()
```
- 清除优化器中的所有累积梯度。

```python
            img, img_metas = data['img'], data['img_metas']
            gt_bboxes_3d = data['gt_bboxes_3d']
            gt_labels_3d = data['gt_labels_3d']
```
- 从数据批次中提取图像、图像元数据、3D边界框和3D标签。

```python
            reference_points_cam, bev_mask, patch_size = self.get_reference_points(img, img_metas, gt_bboxes_3d)
```
- 使用`get_reference_points`方法获取参考点、BEV掩码和补丁大小。

```python
            if reference_points_cam is None:
                print('Warning: No ground truth')
                continue
```
- 如果没有参考点，这意味着没有地面真实值，所以跳过这个批次。

```python
            inputs, is_placed = self.place_patch(img, img_metas, gt_labels_3d, reference_points_cam, bev_mask, patch_size)
```
- 使用`place_patch`方法将补丁放置在图像上，并获取输入和一个布尔值`is_placed`，表示是否放置了补丁。

```python
            if not is_placed:
                continue
```
- 如果没有放置补丁，则跳过这个批次。

```python
            if self.adv_mode:
                outputs = model(return_loss=False, rescale=True, adv_mode=True, **inputs)
            else:
                outputs = model(return_loss=False, rescale=True, **inputs)
```
- 根据`adv_mode`属性，将输入传递给模型并获取输出。

```python
            assign_results = self.assigner.assign(outputs, gt_bboxes_3d, gt_labels_3d)
```
- 使用`assigner`属性将模型的输出分配给地面真实值。

```python
            if assign_results is None:
                print('Warning: No assign results')
                key = list(outputs[0].keys())[0]
                if outputs[0][key]['scores_3d'].numel() != 0:
                    outputs[0][key]['scores_3d'].sum().backward()
                continue
```
- 如果没有分配结果，这可能是因为模型没有预测任何输出或预测不匹配地面真实值。在这种情况下，清除计算图并跳过这个批次。

```python
            loss_adv = -1 * self.loss_fn(**assign_results)
```
- 使用分配的结果计算对抗损失。

```python
            loss_adv.backward()
```
- 对损失进行反向传播，以计算关于补丁的梯度。

```python
            self.optimizer.step()
```
- 使用优化器更新补丁。

```python
            self.patches.data.clamp_(self.lower.view(1, 3, 1, 1), self.upper.view(1, 3, 1, 1))
```
- 使用`clamp_`方法确保补丁的值在`lower`和`upper`之间。

```python
            lr = self.optimizer.param_groups[0]['lr']
            print(f'[Epoch: {i}/{self.epoch}] Iteration: {batch_id}/{len(self.loader)}  Loss: {loss_adv}  lr: {lr}')
```
- 打印当前的迭代信息，包括损失和学习率。

```python
    return self.patches
```
- 返回训练后的补丁。

当然，接下来我将解释`run`和`place_patch`方法。


## 补丁攻击运行

### `run` 方法:

这个方法的目的是在给定的图像上实时放置补丁。

```python
def run(self, model, img, img_metas, gt_bboxes_3d, gt_labels_3d):
```
- 方法接受五个参数：模型、图像、图像元数据、3D边界框和3D标签。

```python
    reference_points_cam, bev_mask, patch_size = self.get_reference_points(img, img_metas, gt_bboxes_3d)
```
- 使用`get_reference_points`方法获取参考点、BEV掩码和补丁大小。

```python
    if reference_points_cam is None:
        return {'img': img, 'img_metas': img_metas}
```
- 如果没有参考点，这意味着没有地面真实值，所以返回原始图像和其元数据。

```python
    else:
        inputs, _ = self.place_patch(img, img_metas, gt_labels_3d, reference_points_cam, bev_mask, patch_size)
        return inputs
```
- 否则，使用`place_patch`方法将补丁放置在图像上，并返回处理后的输入。


## 补丁放置

### `place_patch` 方法:

这个方法的目的是将补丁放置在图像的中心。

```python
def place_patch(self, img, img_metas, gt_labels, reference_points_cam, bev_mask, patch_size=torch.tensor((5,5))):
```
- 方法接受六个参数：图像、图像元数据、地面真实标签、参考点、BEV掩码和补丁大小。

```python
    img_copy = deepcopy(img)
```
- 创建图像的深拷贝，以便在其上放置补丁而不影响原始图像。

```python
    img_ = img_copy[0].data[0].clone()
    gt_labels = gt_labels[0].data[0][0]
```
- 提取图像数据和地面真实标签。

```python
    if self.mono_model:
        B, C, H, W = img_.size()
        M = 1
    else:
        B, M, C, H, W = img_.size()
```
- 根据是否是单目模型来获取图像的尺寸。

```python
    M_, N = reference_points_cam.size()[:2]
```
- 获取参考点的尺寸。

```python
    assert M == M_, f"camera number in image({M}) not equal to camera number in reference_points_cam(f{M_})"
```
- 确保图像中的摄像机数量与参考点中的摄像机数量相同。

```python
    assert B == 1, f"Batchsize should be set to 1 when attack, now f{B}"
```
- 确保批次大小为1，因为在攻击时只处理一张图像。

```python
    assert patch_size.size(-1) == 2, f"Last dim of patch size should have size of 2, now f{patch_size.size(0)}"
```
- 确保补丁大小的最后一个维度为2，表示补丁的高度和宽度。

```python
    if not self.dynamic_patch:
        patch_size = patch_size.view(1, 1, 2).repeat(M, N, 1)
```
- 如果不使用动态补丁大小，则将补丁大小重复以匹配参考点的数量。

```python
    patches = self.patches
```
- 获取当前的补丁。

```python
    if not self.category_specify:
        assert patches.size(0) == 1, ...
        patches = patches.repeat(10, 1, 1, 1)
```
- 如果不为每个类别指定补丁，则重复补丁以匹配类别的数量。

```python
    patch_size = torch.div(patch_size, 2, rounding_mode='floor')
```
- 将补丁大小除以2，以获取补丁的一半大小。

```python
    bev_mask = bev_mask.squeeze()
```
- 压缩BEV掩码的维度。

```python
    neg_x = torch.maximum(reference_points_cam[..., 0] - patch_size[..., 0], torch.zeros_like(reference_points_cam[..., 0])) * bev_mask
    ...
```
- 计算补丁在图像上的位置。

```python
    is_placed = False
    for m in range(M):
        for n in range(N):
            ...
            img_[0, m, :, neg_y[m, n] : pos_y[m, n], neg_x[m, n] : pos_x[m, n]] = resize_patch
            is_placed = True
```
- 在图像上放置补丁，并设置`is_placed`为True。

```python
    img_copy[0].data[0] = img_
    return {'img': img_copy, 'img_metas': img_metas}, is_placed
```
- 返回放置了补丁的图像和其元数据，以及一个布尔值表示是否放置了补丁。

好的，接下来我们将继续解释`get_reference_points`和`get_patch_size`方法。

### `get_reference_points` 方法:

这个方法的目的是从给定的图像、图像元数据和3D边界框中获取参考点。

```python
def get_reference_points(self, img, img_metas, gt_bboxes_3d):
```
- 方法接受三个参数：图像、图像元数据和3D边界框。

```python
    img_ = img[0].data[0].clone()
    B = img_.size(0)
```
- 提取图像数据并获取批次大小。

```python
    assert B == 1, f"When attack models, batchsize should be set to 1, but now {B}"
```
- 确保批次大小为1，因为在攻击时只处理一张图像。

```python
    C, H, W = img_.size()[-3:]
    gt_bboxes_3d_ = gt_bboxes_3d[0].data[0][0].clone()
```
- 获取图像的尺寸和3D边界框数据。

```python
    if len(gt_bboxes_3d_.tensor) == 0:
        return None, None, None
```
- 如果没有3D边界框，返回None。

```python
    center = deepcopy(gt_bboxes_3d_.gravity_center)
    corners = deepcopy(gt_bboxes_3d_.corners)
```
- 从3D边界框中提取中心和角点。

```python
    if self.mono_model:
        center, corners = self.camera2lidar(center, corners, img_metas)
```
- 如果是单目模型，将中心和角点从摄像机坐标转换为激光雷达坐标。

```python
    center = torch.cat(
        (center, torch.ones_like(center[..., :1])), -1).unsqueeze(dim=-1)
```
- 将中心与一个全为1的张量连接，以便进行坐标转换。

```python
    lidar2img = img_metas[0].data[0][0]['lidar2img']
    lidar2img = np.asarray(lidar2img)
    lidar2img = center.new_tensor(lidar2img).view(-1, 1, 4, 4)
```
- 从图像元数据中获取从激光雷达到图像的转换矩阵，并将其转换为张量。

```python
    reference_points_cam = torch.matmul(lidar2img.to(torch.float32),
                                        center.to(torch.float32)).squeeze(-1)
```
- 使用转换矩阵将中心从激光雷达坐标转换为摄像机坐标。

```python
    eps = 1e-5
    bev_mask = (reference_points_cam[..., 2:3] > eps)
```
- 创建一个BEV掩码，用于确定哪些参考点在摄像机坐标中有正的z值。

```python
    reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
        reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)
```
- 将参考点的x和y坐标除以z坐标，以得到它们在图像平面上的位置。

```python
    reference_points_cam = (reference_points_cam * bev_mask * torch.tensor((W, H))).int()
```
- 使用BEV掩码调整参考点，并将其转换为整数坐标。

```python
    if self.dynamic_patch:
        patch_size = self.get_patch_size(corners, lidar2img, bev_mask, scale=self.scale)
    else:
        patch_size = self.patch_size
```
- 如果使用动态补丁大小，则使用`get_patch_size`方法获取补丁大小，否则使用预定义的补丁大小。

```python
    return reference_points_cam, bev_mask, patch_size
```
- 返回参考点、BEV掩码和补丁大小。

### `get_patch_size` 方法:

这个方法的目的是根据在投影图像上的对象大小计算补丁大小。

```python
def get_patch_size(self, corners, lidar2img, bev_mask, scale=0.5):
```
- 方法接受四个参数：角点、从激光雷达到图像的转换矩阵、BEV掩码和缩放因子。

```python
    N, P = corners.size()[:2]
    M = lidar2img.size(0)
```
- 获取角点的尺寸和转换矩阵的尺寸。

```python
    corners = torch.cat(
        (corners, torch.ones_like(corners[..., :1])), -1).unsqueeze(dim=-1)
```
- 将角点与一个全为1的张量连接，以便进行坐标转换。

```python
    corners = corners.view(N*8, 4, 1)
    img_corners = torch.matmul(lidar2img.to(torch.float32),
                               corners.to(torch.float32)).squeeze(-1)
```
- 将角点重塑并使用转换矩阵将它们从激光雷达坐标转换为摄像机坐标。

```python
    img_corners = img_corners.view(M, N, 8, 4)
    eps = 1e-5
    img_corners = img_corners[..., 0:2] / torch.maximum(
        img_corners[..., 2:3], torch.ones_like(img_corners[..., 2:3]) * eps)
```
- 将角点重塑并将其x和y坐标除以z坐标，以得到它们在图像平面上的位置。

```python
    img_corners = img_corners * bev_mask.view(M, N, 1, 1)
    xmax = img_corners[..., 0].max(dim=-1)[0]
    ...
```
- 使用BEV掩码调整角点，并计算补丁的最大和最小坐标。

```python
    patch_size = torch.zeros((M, N, 2))
    patch_size[..., 0] = (scale * (xmax - xmin))
    patch_size[..., 1] = (scale * (ymax - ymin))
```
- 创建一个补丁大小张量，并根据最大和最小

坐标以及给定的缩放因子计算补丁的宽度和高度。

```python
    return patch_size
```
- 返回补丁大小。

这就是`get_reference_points`和`get_patch_size`方法的逐行解释。这些方法主要用于确定在图像上放置补丁的位置和大小。


当然，我们将继续解释剩余的方法。

### `camera2lidar` 方法:

这个方法的目的是将从摄像机坐标转换到激光雷达坐标。

```python
def camera2lidar(self, center, corners, img_metas):
```
- 方法接受三个参数：中心、角点和图像元数据。

```python
    assert 'sensor2lidar_translation' in list(img_metas[0].data[0][0].keys())
    assert 'sensor2lidar_rotation' in list(img_metas[0].data[0][0].keys())
```
- 确保图像元数据中包含转换和旋转信息。

```python
    sensor2lidar_translation = np.array(img_metas[0].data[0][0]['sensor2lidar_translation'])
    sensor2lidar_rotation = np.array(img_metas[0].data[0][0]['sensor2lidar_rotation'])
```
- 从图像元数据中提取转换和旋转信息。

```python
    center = center @ sensor2lidar_rotation.T + sensor2lidar_translation
    corners = corners @ sensor2lidar_rotation.T + sensor2lidar_translation
```
- 使用提取的旋转和转换信息将中心和角点从摄像机坐标转换到激光雷达坐标。

```python
    return center, corners
```
- 返回转换后的中心和角点。

### `_adjust_learning_rate` 方法:

这个方法的目的是根据给定的步骤调整学习率。

```python
def _adjust_learning_rate(self, step):
```
- 方法接受一个参数：步骤。

```python
    lr = self.lr * np.cos(step / self.max_train_samples * np.pi / 2)
```
- 使用余弦退火策略计算新的学习率。

```python
    for param_group in self.optimizer.param_groups:
        param_group["lr"] = lr
```
- 更新优化器中的学习率。

这就是`camera2lidar`和`_adjust_learning_rate`方法的逐行解释。这些方法主要用于坐标转换和学习率调整。