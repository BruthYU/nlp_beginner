|  Model   | Score | Rank|
|  ----  | ---- |----|
| CNN  | 0.59483 |495/860|
| RNN  | 0.61873 |328/860|

### 1.Simple Evaluaion
Create checkpoint folder
```bash
cd .\RNN\model
mkdir checkpoints
```
Download the pretrained weights

```bash
Link:https://pan.baidu.com/s/1UhK3Y5bkQvqQzyrKst89eg
Code:8k2f
```
Evaluate
```bash
python evaluate.py
```

### 2.Train
Configuration
```bash
.\CNN\model\config.py
```

Train
```bash
python train.py
```