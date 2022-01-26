# VITS
## 概要
PytorchによるVITSの実装です。  
日本語音声のデータセット"<a href="https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus">JVS corpus</a>"で学習し、音声間の変換とText-to-Speechを行うことができます。  

## 想定環境
Ubuntu20.04  
Python 3.8.5, torch==1.10.1+cu113, torchaudio==0.10.1+cu113  
Cython==0.29.26  
ライブラリの詳細は`requirements.txt`を参照。

## プログラム
- `jvs_preprocessor.py`はJVS corpusに対し前処理を行うプログラムです。  
- `vits_train.py`は前処理済みデータセットを読み込み学習を実行し、学習の過程と学習済みパラメーターを出力するプログラムです。  
- `vits_voice_converter.py`は`vits_train.py`で出力した学習済みパラメーターを読み込み、推論(音声間の変換)を実行、結果を`.wav`形式で出力するプログラムです。  
- `vits_text_to_speech.py`は`vits_train.py`で出力した学習済みパラメーターを読み込み、推論(テキストから音声の生成)を実行、結果を`.wav`形式で出力するプログラムです。  

## 使い方

### データセットの用意
1. <a href="https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus">JVS corpus</a>をダウンロード、解凍します。  
2. `jvs_preprocessor.py`の16行目付近の変数`jvs_dataset_path`で、解凍したJVS corpusへのパスを指定します。  
3. `python jvs_preprocessor.py`を実行し前処理を実行します。  
    * データセット中の各`.wav`ファイルがサンプリングレート16000[Hz]へと変換され、`./dataset/jvs_preprocessed/jvs_wav_preprocessed/`以下に出力されます。  
    * 前処理済み各`.wav`ファイルへのパスと、それに対応するラベルが列挙されたファイルが`./dataset/jvs_preprocessed/jvs_preprocessed_for_train.txt`として出力されます。 

### Cythonのモジュールのコンパイル
モジュール`monotonic_align`は高速化のためCythonで実装されています。これをコンパイルします。  
1. `cd ./module/model_component/monotonic_align/`を実行します。  
2. `python setup.py build_ext --inplace`でCythonで書かれたモジュールのコンパイルを行います。  

### 学習
1. `python vits_train.py`を実行しVITSの学習を行います。 
    * 学習過程が`./output/vits/train/`以下に出力されます。  
    * 学習済みパラメーターが`./output/vits/train/iteration295000/netG_cpu.pth`などという形で5000イテレーション毎に出力されます。  

### 推論(音声変換)
1. `vits_voice_converter.py`の37行目付近の変数`trained_weight_path`に`vits_train.py`で出力した学習済みパラメーターへのパスを指定します。  
2. `vits_voice_converter.py`の39行目付近の変数`source_wav_path`に変換元としたいwavファイルへのパスを指定します。  
3. `vits_voice_converter.py`の41行目付近の変数`source_speaker_id`に変換元の話者idを指定します。  
    * 話者idは(JVS corpusで指定されている話者の番号-1)となります。例えば"jvs010"の話者を指定したい場合は、話者idは9となります。  
4. `vits_voice_converter.py`の43行目付近の変数`target_speaker_id`に変換先の話者idを指定します。  
5. `python vits_voice_converter.py`を実行し推論(音声変換)を行います。  
    * 変換結果が`./output/vits/inference/voice_conversion/output.wav`として出力されます。  

### 推論(Text-to-Speech)
1. `vits_text_to_speech.py`の39行目付近の変数`trained_weight_path`に`vits_train.py`で出力した学習済みパラメーターへのパスを指定します。  
2. `vits_text_to_speech.py`の41行目付近の変数`source_text`に発話させたい文章を指定します。  
4. `vits_text_to_speech.py`の43行目付近の変数`target_speaker_id`に発話の対象とする話者idを指定します。  
5. `python vits_text_to_speech.py`を実行しText-to-Speechを行います。  
    * 生成結果が`./output/vits/inference/text_to_speech/output.wav`として出力されます。  

## 参考
<a href="https://arxiv.org/abs/2106.06103">https://arxiv.org/abs/2106.06103</a>  
<a href="https://github.com/jaywalnut310/vits">https://github.com/jaywalnut310/vits</a>  
