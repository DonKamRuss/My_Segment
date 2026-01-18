# 1. Принудительный импорт и установка области видимости
custom_imports = dict(
    imports=['mmseg.datasets', 'mmseg.models'],
    allow_failed_imports=False)

default_scope = 'mmseg' # Обязательно для корректной работы реестра

# Базовые настройки
_base_ = [
    '../_base_/models/segformer_mit-b0.py',
    '../_base_/default_runtime.py'
]

# 2. Параметры данных
dataset_type = 'StudDataset' 
data_root = 'E:/Yandex_prakt/1_Proj_segment/stud_dataset/'
crop_size = (256, 256)

metainfo = dict(
    classes=('background', 'cat', 'dog'),
    palette=[[120, 120, 120], [6, 230, 230], [200, 50, 50]]
)

# 3. Модель и Лоссы
model = dict(
    data_preprocessor=dict(size=crop_size),
    decode_head=dict(
        num_classes=3,
        loss_decode=[
            dict(
                type='CrossEntropyLoss', 
                use_sigmoid=False, 
                loss_weight=1.0,
                avg_non_ignore=True,
                class_weight=[0.8, 1.0, 1.0]), # Веса из EDA [cite: 3]
            dict(type='DiceLoss', loss_weight=1.0) # Для mDice
        ]
    )
)

# 4. Пайплайны
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=crop_size, keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

# 5. Загрузчики (CPU оптимизация)
train_dataloader = dict(
    batch_size=2,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='images/train', seg_map_path='annotations/train'),
        metainfo=metainfo,
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='images/val', seg_map_path='annotations/val'),
        metainfo=metainfo,
        pipeline=test_pipeline))

test_dataloader = val_dataloader

# 6. Метрики и Обучение
val_evaluator = dict(type='IoUMetric', iou_metrics=['mDice', 'mIoU'])
test_evaluator = val_evaluator

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0006, weight_decay=0.01))

param_scheduler = [
    dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=500),
    dict(type='PolyLR', eta_min=0.0, power=1.0, begin=500, end=10000, by_epoch=False)
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=10000, val_interval=1000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# [cite_start]7. Визуализация и ClearML [cite: 4, 5, 6]
visualizer = dict(
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'), # Теперь будет использовать директорию по умолчанию из runtime
        dict(
            type='ClearMLVisBackend',
            init_kwargs=dict(
                project_name='Segment',
                task_name='segformer-b0-final-fix'
            )
        )
    ],
    save_dir='work_dirs/visualizations' # Явное указание пути
)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000, max_keep_ckpts=2),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=1),
    # Добавляем раннюю остановку
    early_stopping=dict(
        type='EarlyStoppingHook',
        monitor='mDice',      # Метрика, за которой следим
        rule='greater',       # Ждем увеличения метрики
        patience=2,           # Сколько валидаций ждать без улучшений (2 * val_interval)
        strict=False
    )
)