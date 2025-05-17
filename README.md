# задача сформировать представления о проектах c CV и ML 
- сформировать список библиотек с которомы можно работать на CPU 

cv2 opencv 
torch pythorch

- сгенерировать тестовые данные
  генерация данных может сильно влиять на процесс обучения сети , можно использовать dxf контуры для специалиных обьектов
  скорость генерации сильно растёт с модулем concurrent или на потоках или асинхронно
- структуру проектов 
  <div style="font-family: monospace; white-space: pre;">
project_root/
├── data/                   
│&nbsp;&nbsp;├── raw/                
│&nbsp;&nbsp;│&nbsp;&nbsp;├── images/         
│&nbsp;&nbsp;│&nbsp;&nbsp;└── dxf_templates/  
│&nbsp;&nbsp;├── processed/          
│&nbsp;&nbsp;│&nbsp;&nbsp;├── train/          
│&nbsp;&nbsp;│&nbsp;&nbsp;│&nbsp;&nbsp;├── images/     
│&nbsp;&nbsp;│&nbsp;&nbsp;│&nbsp;&nbsp;└── masks/      
│&nbsp;&nbsp;│&nbsp;&nbsp;├── val/            
│&nbsp;&nbsp;│&nbsp;&nbsp;└── test/           
│&nbsp;&nbsp;└── annotations/        
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── coco/           
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── yolo/           
├── src/                    
│&nbsp;&nbsp;├── data_preprocessing/ 
│&nbsp;&nbsp;│&nbsp;&nbsp;├── dxf_to_mask.py  
│&nbsp;&nbsp;│&nbsp;&nbsp;└── augmentations.py
│&nbsp;&nbsp;├── modeling/           
│&nbsp;&nbsp;│&nbsp;&nbsp;├── train.py        
│&nbsp;&nbsp;│&nbsp;&nbsp;├── inference.py    
│&nbsp;&nbsp;│&nbsp;&nbsp;└── utils/          
│&nbsp;&nbsp;└── visualization/      
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── plot_results.py 
├── models/                 
│&nbsp;&nbsp;├── checkpoint.pth      
│&nbsp;&nbsp;└── opencv_templates/   
├── configs/                
│&nbsp;&nbsp;├── train_config.yaml   
│&nbsp;&nbsp;└── data_config.yaml    
├── notebooks/              
│&nbsp;&nbsp;└── EDA.ipynb           
├── tests/                  
│&nbsp;&nbsp;└── test_preprocessing.py
├── requirements.txt        
├── README.md              
└── .gitignore             
</div>

# запуск, обслуживание, контроль

- докеризация?
- реальное железо?
  использование процессоров с avx2 минимум для opencv cv2, использование потоков ускоряет процессы в 10ки раз