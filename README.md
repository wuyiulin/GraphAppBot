# GraphAppBot

## Introduction
Telegram bot for detecting image modifications.

The detection principle behind it is derived from the implementation of ['Analyzing Benford’s Law’s Powerful Applications in Image Forensics'](https://www.mdpi.com/2076-3417/11/23/11482) paper.

## How to use?

Simply send an image to the bot.

## Example

![image](https://github.com/wuyiulin/GraphAppBot/blob/main/Photo/example.jpg)

## Analyze

If the "修圖程度" of editing exceeds 0.6, it can be considered as edited.

P.S. Since the detection principle relies on brightness data, the native white balance in the camera is also considered as a form of editing.


## 簡介

這是一個 檢測修圖程度的 Telegram Bot。

檢測原理來自於這篇論文 ['Analyzing Benford’s Law’s Powerful Applications in Image Forensics'](https://www.mdpi.com/2076-3417/11/23/11482)

## 用法

直接傳圖片給 Telegram 機器人就可以了 >.0

## 範例

![image](https://github.com/wuyiulin/GraphAppBot/blob/main/Photo/example.jpg)

## 分析

如果檢測到的修圖程度數值超過 0.6，可以認定為有修圖。

PS.因為檢測原理依賴亮度數據，所以相機內原生白平衡也算修圖的一種。

## murmur

這邊還有兩個問題要解決：

1.目前版本用的是 OpenCV 的 DCT ，比我自造的 DCT 要快不少，我想這應該是 OpenCV 用矩陣預存 8x8 的 DCT 結果，有時間可以改進。

2.我有寫了一版 Muti Threads 的 DCT，但是會在記憶體上面遇到 Race Condition 的問題，再想想 Python 上面能怎麼做。


如果有同道中人願意一起探討，或是我的論文實現哪裡有問題？

麻煩聯絡我　wuyiulin@gmail.com

感激不盡

