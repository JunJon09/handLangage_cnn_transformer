FROM python:3.9.7

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
# MeCabとその依存関係のインストール
RUN apt-get update && apt-get install -y libgl1-mesa-dev
COPY . .
