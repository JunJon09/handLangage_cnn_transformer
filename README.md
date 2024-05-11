<div id="top"></div>

## 使用技術一覧

<!-- シールド一覧 -->
<!-- 該当するプロジェクトの中から任意のものを選ぶ-->
<p style="display: inline"> 
  <!-- バックエンドの言語一覧 -->
  <img src="https://img.shields.io/badge/-Python-F2C63C.svg?logo=python&style=for-the-badge">
</p>

## 目次

1. [プロジェクトについて](#プロジェクトについて)
2. [環境](#環境)
3. [ディレクトリ構成](#ディレクトリ構成)
4. [開発環境構築](#開発環境構築)



<!-- プロジェクト名を記載 -->

## プロジェクト名

handLangage_cnn_transformer

<!-- プロジェクトについて -->

## プロジェクトについて

3DCNNとTransformerを使用した手話認識


<p align="right">(<a href="#top">トップへ</a>)</p>

## 環境

<!-- 言語、フレームワーク、ミドルウェア、インフラの一覧とバージョンを記載 -->

| 言語・フレームワーク  | バージョン |
| --------------------- | ---------- |
| Python                | 3.9.7     |


その他のパッケージのバージョンは requirements.txt を参照してください

<p align="right">(<a href="#top">トップへ</a>)</p>

## ディレクトリ構成

<!-- Treeコマンドを使ってディレクトリ構成を記載 -->

.

├── data
├── .gitignore
├── LSA64
    ├──all
├── README.md
├── modules
    ├── datafix.py
    ├── LSA64_split.py
    ├──machineLearning.py
    ├──preTrainingData3DCNN.py
    ├──preTrainingDataTransformer.py
    ├──transformer_model.pth
    ├──video_3dcnn_model.pth
├── test
│   ├── pre
│   ├── test
│   ├── train
│


<p align="right">(<a href="#top">トップへ</a>)</p>

## 開発環境構築

pip install -r requirements.txt 
Windowsを使用する場合は、Microsoft Visual C++ Build Tools のインストール(https://visualstudio.microsoft.com/downloads/)してもらい、Windows 10 SDKと、最新のMSVCのパッケージも入れる



