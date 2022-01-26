# VITS
## 概要
PytorchによるVITSの実装です。  
データセット"<a href="https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus">JVS corpus</a>"で学習し、音声間の変換とText-to-Speechを行うことができます。  

## 想定環境
Ubuntu20.04  
Python 3.8.5, torch==1.10.1+cu113, torchaudio==0.10.1+cu113  
ライブラリの詳細は`requirements.txt`を参照。

## プログラム
- `jvs_preprocessor.py`はJVS corpusに対し前処理を行うプログラムです。  
- `vits_train.py`は前処理済みデータセットを読み込み学習を実行し、学習の過程と学習済みパラメーターを出力するプログラムです。  
- `vits_voice_converter.py`は`vits_train.py`で出力した学習済みパラメーターを読み込み、推論(音声間の変換)を実行、結果を`.wav`形式で出力するプログラムです。  
- `vits_text_to_speech.py`は`vits_train.py`で出力した学習済みパラメーターを読み込み、推論(テキストから音声の生成)を実行、結果を`.wav`形式で出力するプログラムです。  

## 使い方

## 参考
<a href="https://arxiv.org/abs/2106.06103">https://arxiv.org/abs/2106.06103</a>  
<a href="https://github.com/jaywalnut310/vits">https://github.com/jaywalnut310/vits</a>  
