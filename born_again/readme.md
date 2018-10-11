# KerasでBorn-Again Neural Networks

タイトル通り、[Born-Again Neural Networks](https://arxiv.org/abs/1805.04770)の再現実装です（ただしネットワークは論文よりシンプルなもの）。
  
```./original_network.py```で学習した後、```./born_again.py```で保存したモデルを教師モデルとして学習を進めることができます。  

### 学習の実行  
```
python born_again.py --temperature 10 --lambda_const 0.9 --teacher_model_path ./models/$hoge
```
### 引数について  
temperatureはsoft targetを計算する際の温度項、lambda_constはhard lossとsoft lossの比です。(
[Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531)を参照）

### 補足  
ここでは```original_network.py```と```born_again.py```のネットワークが同一のためBorn-Againとなっていますが、このネットワークを変更すれば通常のKnowledge Distillationを行うことも出来ます。
```
