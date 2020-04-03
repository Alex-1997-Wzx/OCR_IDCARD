# -- coding: utf-8 --
# @Time : 2020/3/28 下午6:03
# @Author : Gao Shang
# @File : upload_images.py
# @Software : PyCharm

import flask
from flask import Flask, render_template, request, redirect, url_for, make_response, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import time
from datetime import timedelta
from recognition_words_module.model import networks
from detection_card_module import detection_card
from detection_text_module import detection_text
from recognition_words_module import recognition_words
import torch
from recognition_words_module.alphabet import alphabet


# 设置允许的文件格式
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])

absolute_path = os.path.dirname(__file__)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


app = Flask(__name__)
# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=1)

model = None


def load_model():
    global model
    model = [networks.chsNet(1, len(alphabet[0]) + 1), networks.digitsNet(1, len(alphabet[1]) + 1)]
    if torch.cuda.is_available():
        # 中文模型
        model[0] = model[0].cuda()
        # 数字模型
        model[1] = model[1].cuda()
    # 加载预训练模型
    model[0].load_state_dict({k.replace('module.', ''): v for k, v in
                              torch.load('../recognition_words_module/data/chs.pth').items()})
    model[1].load_state_dict({k.replace('module.', ''): v for k, v in
                              torch.load('../recognition_words_module/data/number.pth').items()})


@app.route('/', methods=['POST', 'GET'])  # 添加路由
def upload():
    if request.method == 'POST':
        f = request.files['file']

        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})

        current_path = os.path.dirname(__file__)  # 当前文件所在路径

        # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
        upload_path = os.path.join(current_path, 'static/image_input', 'test.jpg')
        f.save(upload_path)

        image = cv2.imread(upload_path, 0)
        points = detection_card.getCardPoint(image)
        text = detection_text.getTextLine(image, points)
        result = recognition_words.getWordsResult(text, model)

        return render_template('recognition.html', words=result, filename=secure_filename(f.filename))

    return render_template('index.html')


if __name__ == '__main__':
    # app.debug = True
    load_model()
    app.run(debug=True)

