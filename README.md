# 基于OCR的身份证要素提取设计与实现

## 1.项目背景：[基于OCR识别技术的身份证要素提取](https://www.datafountain.cn/competitions/346)

## 2.环境版本说明：
	- Anaconda3: 4.7.12
	- PyCharm: 2019.3.3
	- OpenCV-Python: 4.2.0.32
	- Numpy: 1.18.1
	- Flask: 1.1.1
	- 详细组件版本信息请查看conda list文件

## 3.模块说明：
    - detection_card_module: 定位复印件中的身份证顶点坐标
    - detection_text_module: 定位身份证图片中的文本行信息
    - recognition_words_module: 识别文本行信息
    - web-module: 提供可视化界面
    - train_model_module: 训练模型代码

## 4.使用说明：
字符界面：python ./main.py
可视化界面：python ./web_module/server.py 访问：http://127.0.0.1

## 4.作者：
内蒙古师范大学－计算机科学技术学院－高尚

## 5.版权说明：
个人项目部分代码，欢迎指正讨论。