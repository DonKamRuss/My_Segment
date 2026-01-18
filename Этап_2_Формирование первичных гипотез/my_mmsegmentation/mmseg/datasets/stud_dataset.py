from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


# Этот декоратор нужен, чтобы зарегистрировать наш класс в системе mmsegmentation.
# Это позволит ссылаться на этот класс в конфигах 
@DATASETS.register_module()
class StudDataset(BaseSegDataset):
    # Поскольку наши данные уже организованы в формате для BaseSegDataset, 
    # нам остаётся только добавить общую метаинформаиию:
    # 1. список классов 
    # 2. цветовая палитра 
    
    metainfo = dict(
    classes=('background', 'cat', 'dog'),
    palette=[[120, 120, 120], [6, 230, 230], [200, 50, 50]]
)

    # Конструктор оставляем, как был
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) 