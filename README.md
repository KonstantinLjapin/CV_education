# задача сформировать представления о проектах c CV и ML 
- сформировать список библиотек с которомы можно работать на CPU 
- сгенерировать тестовые данные
генерация данных может сильно влиять на процесс обучения сети , можно использовать dxf контуры для специалиных обьектов
скорость генерации сильно растёт с модулем concurrent или на потоках или асинхронно
- структуру проектов
<pre> project_root/ ├── data/ │ ├── raw/ │ │ ├── images/ │ │ └── dxf_templates/ │ ├── processed/ │ │ ├── train/ │ │ │ ├── images/ │ │ │ └── masks/ │ │ ├── val/ │ │ └── test/ │ └── annotations/ │ ├── coco/ │ └── yolo/ ├── src/ │ ├── data_preprocessing/ │ │ ├── dxf_to_mask.py │ │ └── augmentations.py │ ├── modeling/ │ │ ├── train.py │ │ ├── inference.py │ │ └── utils/ │ └── visualization/ │ └── plot_results.py ├── models/ │ ├── checkpoint.pth │ └── opencv_templates/ ├── configs/ │ ├── train_config.yaml │ └── data_config.yaml ├── notebooks/ │ └── EDA.ipynb ├── tests/ │ └── test_preprocessing.py ├── requirements.txt ├── README.md └── .gitignore </pre>
- запуск, обслуживание, контроль

- докеризация?
- реальное железо?

запуск python cross_center_detected.py