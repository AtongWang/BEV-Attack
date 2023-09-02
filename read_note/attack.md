# 启动命令

根据[attack.py](mmdet_adv/tools/attack.py)中的args，可得到相应输入为：
```
python attack.py $config $checkpoint $wb_checkpoint $out $save-results


'config', help='test config file path'
'checkpoint', help='checkpoint file'
'--wb_checkpoint', help='white box checkpoint for transfer black box attack'
'--out', help='output result file in pickle format'
'--save-results', help='output result file in pickle format'

```
[attack.sh](mmdet_adv/tools/attack.sh)中命令如下：

```
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python ./tools/attack.py \ 
/home/cixie/shaoyuan/BEV-Attack/mmdet_adv/projects/configs/attack/detr3d_wo_cbgs.py \ config
/home/cixie/shaoyuan/BEV-Attack/models/detr3d/detr3d_resnet101.pth \  checkpoint
--out pgd_new \ out
```

# Attack.py 分析

这段代码是一个主程序 (`main()`)，主要处理攻击模型的任务。我将会逐步解析这段代码的核心部分和它们的作用。

## **解析命令行参数**:

```python
args = parse_args()
assert args.out is not None, "Specify output dir by add --out argument"
```

通过 `parse_args()` 函数获取命令行参数，并确保用户已经指定了输出目录 (`args.out`)。

##  **加载配置文件**:

```python
cfg = Config.fromfile(args.config)
```

通过 `args.config` 指定的路径，使用 `Config.fromfile` 加载配置文件。

Note:本质就是基于Dict的角度取存储、调度、利用程序所需的各种变量

##  **导入自定义模块**:

如果配置 (`cfg`) 中指定了 `custom_imports`，那么会从这些字符串列表中导入模块。

Note:该命令`from mmcv.utils import import_modules_from_strings`，`import_modules_from_strings`可以直接基于str进行库的相关调用

##  **插件模块导入**:

如果配置有插件信息，代码将导入这些插件模块。

`plugin_dir='projects/mmdet3d_plugin/` 转化为`projects.mmdet3d_plugin` 然后利用 importlib 将str-》module

```python
if hasattr(cfg, 'plugin'):
    if cfg.plugin:
        import importlib
        if hasattr(cfg, 'plugin_dir'):
            if isinstance(cfg.plugin_dir, str):
                cfg.plugin_dir = [cfg.plugin_dir]
            # import multi plugin modules
            for plugin_dir_ in cfg.plugin_dir:
                _module_dir = os.path.dirname(plugin_dir_)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
```

## **确定攻击类型**:

根据给定的参数和配置，代码确定是否执行转移攻击 (`transfer_attack`)。

```
if args.wb_checkpoint is not None or cfg.get('wb_model', None):
    assert cfg.get('wb_model', None), "When activate black box attack, should specify wb_model in config file"
    assert args.wb_checkpoint is not None, "When activate black box attack, should specify wb_model checkpoint in arg"
    transfer_attack = True
```
如果程序输入$wb_checkpoint 为None，或者config无该变量内容，则跳过。
否则说明要进行迁移攻击，所以加入了断言

## **设置 CUDA 的 cuDNN 算法优化**:

如果配置中指定，将 `torch.backends.cudnn.benchmark` 设为 `True`，以提高某些固定尺寸输入的性能。

## **数据处理与加载**:

根据配置信息，处理数据并构建对应的数据加载器。

这段代码与模型测试数据的配置和预处理相关。以下是代码的详细步骤：

1. **关闭预训练模式**:
    ```python
    cfg.model.pretrained = None
    ```
    这行代码是为了确保模型不加载预训练的权重，因为我们通常希望使用特定的检查点来测试模型。

2. **设置默认的`samples_per_gpu`**:
    ```python
    samples_per_gpu = 1
    ```
    默认情况下，每个GPU处理的样本数量为1。

3. **检查`cfg.data.test`的数据结构**:
    - 如果`cfg.data.test`是一个字典，这意味着我们只有一个测试数据集。
    - 如果`cfg.data.test`是一个列表，这意味着我们有多个测试数据集，它们都存储在这个列表中。

4. **处理`cfg.data.test`为字典的情况**:
    ```python
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    ```
    - 设置测试模式为`True`。
    - 尝试从字典中获取`samples_per_gpu`值，如果不存在则使用默认值1。
    - 如果`samples_per_gpu`大于1，则在处理管道中将`ImageToTensor`替换为`DefaultFormatBundle`。

5. **处理`cfg.data.test`为列表的情况**:
    ```python
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max([ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)
    ```
    - 遍历每个测试数据集的配置，并为每个数据集设置测试模式为`True`。
    - 获取所有数据集中的最大`samples_per_gpu`值。
    - 如果`samples_per_gpu`大于1，则在每个数据集的处理管道中将`ImageToTensor`替换为`DefaultFormatBundle`。

6. **定义测试数据集的配置**:
    ```python
    test=dict(type=dataset_type,
              data_root=data_root,
              ann_file=data_root + 'nuscenes_infos_temporal_val.pkl',
              pipeline=test_pipeline, bev_size=(bev_h_, bev_w_),
              test_mode=False,
              adv_mode=True,
              classes=class_names, modality=input_modality),
    ```
    这部分代码定义了一个名为`test`的字典，描述了测试数据集的各种属性，如：
    - 数据集类型 (`type`)。
    - 数据的根目录 (`data_root`)。
    - 注释文件路径 (`ann_file`)。
    - 数据处理管道 (`pipeline`)。
    - BEV (Bird's Eye View) 的大小 (`bev_size`)。
    - 是否为测试模式 (`test_mode`)。
    - 是否为进阶模式 (`adv_mode`)。
    - 数据集中的类别名称 (`classes`)。
    - 输入的模式 (`modality`)。

总之，此代码段主要是对测试数据集的配置和预处理。它首先关闭了模型的预训练模式，然后基于`cfg.data.test`的数据结构对测试数据进行预处理，并最终定义了一个名为`test`的字典，描述了测试数据集的配置。

`DefaultFormatBundle` 怎么理解

> DefaultFormatBundle 是在一些计算机视觉库中用于数据预处理的组件，特别是在MMCV和MMDetection这样的开源库中。当处理图像或目标检测任务的数据时，通常会有一个处理管道，该管道由一系列的转换或操作组成，用于将原始数据转换为模型可以接受的格式。

    以下是DefaultFormatBundle的一些关键点：

    功能:

    它主要负责收集数据并将其格式化为适用于模型训练或测试的格式。
    当在目标检测任务中使用时，它会处理图像、边界框、标签等，并确保它们的形状、数据类型等都是适当的。
    操作:

    对于图像，它可能会确保图像的形状是适当的，并且值的范围是正确的（例如，0到1之间）。
    对于目标（如边界框或标签），它会确保它们的形状和数据类型是适当的，并可能将它们转换为特定的格式（如从列表到张量）。
    为什么需要它:

    当进行计算机视觉任务时，有多种方式可以表示或存储数据。例如，图像可以作为不同类型的数组（如numpy数组或PyTorch张量）存储，边界框可以作为列表、字典或数组存储，等等。
    为了确保模型可以正常工作，输入数据必须具有适当的格式和属性。DefaultFormatBundle确保了这一点。
    替换操作:

    在你之前的代码片段中，当samples_per_gpu大于1时，ImageToTensor被替换为DefaultFormatBundle。这意味着原始的处理流程只涉及将图像转换为张量的操作，而替换后的流程可能会包括更多的操作，确保数据是完全格式化的。
    总之，DefaultFormatBundle是一个数据预处理工具，用于确保数据的格式、形状和数据类型适合后续的模型操作。

怎么理解 如果`samples_per_gpu`大于1，则在每个数据集的处理管道中将`ImageToTensor`替换为`DefaultFormatBundle`。
>
    为了更好地理解这个逻辑，我们首先需要明确samples_per_gpu、ImageToTensor和DefaultFormatBundle的概念。

    samples_per_gpu: 这是指在每次GPU操作中，一次性加载并处理的样本数量。通常情况下，这是批量大小的一个表示。批处理可以帮助加速训练和评估过程，因为可以同时处理多个样本。

    ImageToTensor: 这是一个转换操作，将图像数据从其原始格式（通常是numpy数组）转换为适合深度学习框架（如PyTorch或TensorFlow）的张量格式。

    DefaultFormatBundle: 这是一个更为复杂的转换操作，不仅可以转换图像数据，还可以确保所有与任务相关的数据（如边界框、标签等）都被正确地格式化，以适应特定的深度学习框架。

    现在，让我们理解这个逻辑：

    如果samples_per_gpu是1，那么每次都只有一个样本被加载和处理。在这种情况下，简单的ImageToTensor转换可能就足够了。

    然而，如果samples_per_gpu大于1，意味着我们在每个批次中处理多个样本。在这种情况下，我们可能需要进行更复杂的数据处理和格式化，以确保所有样本都能正确地一起被处理。这就是为什么需要用DefaultFormatBundle替换ImageToTensor的原因。

    简而言之，这种替换的逻辑是基于对数据处理和批处理的需求的理解。当处理多个样本时，可能需要对数据进行更多的处理和格式化，这就是为什么当samples_per_gpu大于1时，会选择使用更为复杂的DefaultFormatBundle而不是简单的ImageToTensor转换。


7. **构建数据集**:

    ```python
    set_random_seed(0, deterministic=False)   #mmdet.apis 负责设置随机种子数以及
    ```
    - `torch.backends.cudnn.deterministic = True` 用于保证每次运行相同的输入时，输出结果是确定的。这是通过禁用cuDNN中的一些非确定性算法来实现的。这对于调试和测试非常有用，但可能会导致一些性能损失。

    - `torch.backends.cudnn.benchmark = False` 用于禁用cuDNN中的自动调整算法，该算法在每次运行时根据硬件条件自动选择最适合的算法。这对于保证结果的一致性非常有用，但可能会导致一些性能损失。

    `deterministic=True`会开启上述选项

    ```python
    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    # test_mode false to return ground truth used in the attack
    # chnage after build the dataset to avoid data filtering
    # an ugly workaround to set test_mode = False
    dataset.test_mode = False
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )
    ```

    `build_dataset`构建书数据类，根据`cfg.data.test`的`type`属性名称，来调用该类。
    同时需要注意，如果是新加（继承）的类，需要首先进行类注册，他们都在`DATASET` 大类下边。

    ```python
    @DATASETS.register_module()
    ```

    然后构建`build_dataloader`构建数据data_loader,本质上其实可以发现，深层次的变量类型均为torch下的类型，只是mmlab对相应变量进行了修饰。

##  **模型构建与加载**:

构建模型，然后从检查点文件加载预训练的权重。

这段代码的主要目的是构建模型并加载预训练的检查点。接下来，我会为你详细分析每一部分的代码：

1. **构建模型**
    ```python
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    ```
   在这里，我们首先将训练配置设为 `None`，这可能是为了确保模型处于评估模式而不是训练模式。随后使用 `build_model` 函数根据配置 `cfg.model` 构建模型。

    ```python
    if transfer_attack:
        wb_model = build_model(cfg.wb_model, test_cfg=cfg.get('test_cfg'))
    ```
   如果 `transfer_attack` 是 `True`，我们还构建另一个模型 `wb_model`。

2. **检查是否支持 float16 模型**
    ```python
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        assert False, "Not support float16 model"
        wrap_fp16_model(model)
    ```
   这部分代码检查是否提供了 `fp16` 配置。如果提供了，它会首先断言 `False`，这意味着代码不支持 `float16` 模型，并且程序会在这里终止。但是，如果没有这个断言，那么代码会将模型转换为 `float16` 格式。

3. **加载检查点**
    ```python
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if transfer_attack:
        wb_checkpoint = load_checkpoint(wb_model, args.wb_checkpoint, map_location='cpu')
    ```
   使用 `load_checkpoint` 函数，我们加载预先训练好的模型权重。如果 `transfer_attack` 是 `True`，我们还加载 `wb_model` 的权重。

4. **为模型设置类信息和调色板**
    ```python
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        model.PALETTE = dataset.PALETTE
    ```
   这部分代码检查检查点中是否包含类信息 (`CLASSES`) 和调色板 (`PALETTE`)。如果包含，它会从检查点中取得这些信息；否则，它会从数据集中取得。

5. **固定模型参数**
    ```python
    for n, p in model.named_parameters():
        p.requires_grad = False
    model = MMDataParallel(model, device_ids=[0])
    ```
   我们遍历模型的所有参数，并将它们的 `requires_grad` 属性设为 `False`，这意味着我们不会对这些参数进行更新或优化。然后，我们使用 `MMDataParallel` 包装模型，使其在指定的设备（例如GPU）上运行。

    ```python
    if transfer_attack:
        for n, p in wb_model.named_parameters():
            p.requires_grad = False
        wb_model = MMDataParallel(wb_model, device_ids=[0])
    else:
        wb_model = model
    ```
   如果 `transfer_attack` 是 `True`，我们对 `wb_model` 执行同样的操作。否则，我们将原模型赋值给 `wb_model`。

总结：
这段代码构建了模型，加载了预训练的检查点，并为模型设置了类信息和调色板。然后它固定了模型的参数，确保这些参数在后续操作中不会改变。最后，它使用 `MMDataParallel` 进行模型的并行化处理。

##  **攻击的严重性**:

根据指定的攻击严重性类型选择对应的严重性列表。

这段代码涉及到对某种配置中的攻击严重性类型及其相关参数的检查。让我们逐行解读它：

1. **提取攻击严重性类型**：
    ```python
    attack_severity_type = cfg.attack_severity_type
    ```
    这行代码从配置对象 `cfg` 中提取 `attack_severity_type`。

2. **确认攻击严重性类型是否在允许的键中**：
    ```python
    assert attack_severity_type in list(cfg.attack.keys()), f"Attack severity type {attack_severity_type} \
        is not a parameters in attack type {cfg.attack.type}"
    ```
    这行代码检查 `attack_severity_type` 是否存在于 `cfg.attack` 的键中。如果不存在，则会引发一个错误，指出该严重性类型不是允许的攻击类型的参数。

3. **提取严重性列表**：
    ```python
    severity_list = cfg.attack[attack_severity_type]
    ```
    这行代码从 `cfg.attack` 中获取对应的严重性列表。

4. **确认严重性列表是否为列表类型**：
    ```python
    assert isinstance(severity_list, List), f"{attack_severity_type} in attack {cfg.attack.type} should be list\
        now {type(severity_list)}"
    ```
    这行代码确保 `severity_list` 是一个列表。如果不是，则会引发一个错误，指出严重性类型在指定的攻击类型中应当是一个列表，但现在是另一种数据类型。

总体来说，这段代码确保提供的攻击严重性类型是有效的，并且与其相关的参数是列表类型。这样的检查有助于确保后续的代码能够正常运行，而不会因为错误的配置或数据类型而出错。

##   **设置日志**:

初始化日志并记录模型配置信息和检查点路径。

这段代码主要关于日志记录，将模型的配置信息和其他相关信息写入一个特定的日志文件中。下面是每一部分的详解：

1. **初始化日志对象**：
    ```python
    logging = Logging_str(osp.join('../log', cfg.model.type, args.out, f"{os.path.splitext(os.path.basename(args.config))[0]}.md"))
    ```
    - 使用 `osp.join` 和其他文件操作函数，构建了日志文件的路径。
    - `cfg.model.type`: 取得模型的类型，作为子目录。
    - `args.out`: 取得输出目录名称。
    - `os.path.splitext(os.path.basename(args.config))[0]`: 获取配置文件的文件名（不带后缀），并用作日志文件的主要名称。
    - `.md`: 说明日志文件是一个 Markdown 文件。
    - 使用这个路径，我们初始化了一个 `Logging_str` 对象，我们可以理解为这是一个用于日志记录的自定义类。

2. **记录模型检查点的来源**：
    ```python
    logging.write(f"Load model checkpoint from {args.checkpoint}")
    ```
    这行代码在日志中记录了模型检查点的来源路径。

3. **记录模型配置**：
    ```python
    logging.write(f"## Model Configuration\n")
    logging.write(f"```")
    logging.write(cfg.pretty_text)
    logging.write(f"```\n")
    ```
    - 这部分代码使用 Markdown 语法来格式化记录模型的配置。
    - 使用 `## Model Configuration` 为记录创建一个二级标题。
    - 使用三个反引号 (```) 开始一个代码块，这在 Markdown 语法中用于格式化代码或配置。
    - `cfg.pretty_text`: 这可能是一个属性或方法，用于获取模型配置的美观文本表示形式。
    - 再次使用三个反引号结束代码块。

总体来说，这段代码创建了一个日志文件，其中包含模型检查点的来源和整齐格式化的模型配置。


##   **攻击过程**:

对于每种参数(步长、缩放大小等)，根据`severity_list`迭代其中的参数设置，并构建相应的攻击，并执行单GPU攻击。如果指定，还会保存结果。

```python
for i in range(len(severity_list)):
    cfg.attack[attack_severity_type] = severity_list[i]
    # build attack
    attacker = build_attack(cfg.attack)

    outputs = single_gpu_attack(model, wb_model, data_loader, attacker)
    if args.save_results:
        mmcv.dump(outputs, args.save_results)
```
这段代码定义了两个函数：`single_gpu_attack` 和 `build_attack`。这两个函数与模型的攻击测试有关。

首先，我将为你解释每个函数，然后概述它们的工作原理。

1. **single_gpu_attack函数**:
    - 这个函数用于在单GPU上执行模型的攻击测试。
    - 它接收以下参数：
        - `model`: 要测试的模型。
        - `wb_model`: 被攻击的白盒模型。
        - `data_loader`: 用于加载测试数据的Pytorch数据加载器。
        - `attacker`: 负责执行攻击的对象。
    - 函数首先将模型和白盒模型设置为评估模式。
    - 如果攻击者有一个加载器并处于训练模式，则会训练一个通用的攻击补丁。
        故，需要分析几类Attacker。
        由于仅有Patch_attack需要train，故该过程仅体现它。

        [**Base_attack**](read_note/base_attack.md)
        [**PGD**](read_note/PGD.md)

        [**Patch**](read_note/Patch.md)

    - 随后，函数通过`data_loader`迭代测试数据，使用`attacker`生成攻击输入，然后在原始模型上进行预测。
    - 函数返回预测结果列表。

2. **build_attack函数**:
    - 这个函数用于构建一个攻击者。
    - 它接收以下参数：
        - `cfg`: 攻击的配置。
    - 函数首先深拷贝配置以防止对原始配置进行任何更改。
    - 它检查是否指定了assigner和loss_fn，并断言这两者都存在。
    - 最后，使用ATTACKER注册表构建并返回攻击者。

接下来的代码段似乎与上述两个函数有关，但它们并不在任何函数内部。这可能是因为代码被拆分或不完整。这部分代码似乎是迭代严重性列表，为每个严重性构建一个攻击者，然后使用`single_gpu_attack`函数执行攻击，并将结果保存（如果指定）。

简而言之，这段代码涉及使用特定的攻击配置对模型进行攻击测试，并保存或返回攻击结果。

##   **评估与日志记录**:

使用预定义的评估方式评估攻击结果，并将结果写入日志。

总的来说，这段代码的主要目的是为了在特定配置下对模型进行攻击并评估其效果。它从命令行获取输入参数，加载配置文件，处理数据，构建并加载模型，进行攻击，并将结果评估并记录到日志中。