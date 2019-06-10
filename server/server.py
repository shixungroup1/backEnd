#
# 参考自：https://blog.csdn.net/weixin_36380516/article/details/80347192 
#      https://zhuanlan.zhihu.com/p/23731819
#
#
from werkzeug.utils import secure_filename
from flask import Flask, render_template, jsonify, request, make_response, send_from_directory, abort
import time
import os
from strUtil import *
import base64
from flask_cors import CORS
import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
from urllib.request import urlretrieve

app = Flask(__name__)

# 解决跨域访问问题
# r'/*' 是通配符，让本服务器所有的URL 都允许跨域请求
CORS(app, supports_credentials=True, resources=r'/*')

UPLOAD_FOLDER = 'upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

basedir = os.path.dirname(os.path.abspath("__file__"))
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'gif', 'GIF', 'mp4', 'jpeg', 'JPEG'])


def allowed_file(filename):
    '''
    可上传文件类型
    filename: 上传文件名称（包括后缀）
    '''
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/upload')
def upload_test():
    '''
    测试代码
    '''
    return render_template('index.html')
 
   
# 上传文件
@app.route('/upload_image', methods=['POST'])
def api_upload():
    '''
    文件上传api
    '''
    global img
    file_dir = os.path.join(basedir, app.config['UPLOAD_FOLDER'])
    if request.method == 'POST':
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        if request.data != b'':
            url_data = str(request.data, encoding='utf-8')
            img_url = url_data[8:-2]
            img_url = str(img_url)
            url_split = img_url.split('/')
            fname = url_split[-1]
            
            if allowed_file(fname):
                urlretrieve(img_url, os.path.join(file_dir, fname))
                print(fname, 'upload success')
                # 成功，返回保存的文件名，以便前端可利用
                return jsonify({"success": 0, "msg": "upload successfully", "name": fname})
            else:
                return jsonify({"error": 1001, "msg": "failed to upload"})
        else:
            f = request.files['file']
            if f and allowed_file(f.filename):
                fname = secure_filename(f.filename)

                # ext = fname.rsplit('.', 1)[1]
                # !!! 返回new_filename，使得前端可以直接访问被重命名的图片
                # new_filename = Pic_str().create_uuid() + '.' + ext

                # 把图片保存在upload文件目录下
                f.save(os.path.join(file_dir, fname))
                print(fname, 'upload success')
                # 成功，返回保存的文件名，以便前端可利用
                return jsonify({"success": 0, "msg": "upload successfully", "name": fname})
            else:
                return jsonify({"error": 1001, "msg": "failed to upload"})
    else:
        pass

    
@app.route('/download/<string:filename>', methods=['GET'])
def download(filename):
    '''
    根据文件名download文件
    filename: 传进来的文件名
    '''
    if request.method == "GET":
        if os.path.isfile(os.path.join('upload', filename)):
            return send_from_directory('upload', filename, as_attachment=True)
        pass
    

# get image
@app.route('/get_image/<string:filename>', methods=['GET'])
def get_image(filename):
    targetFile = os.path.join('upload', filename)
    if request.method == 'GET':
        data = b""
        if os.path.exists(targetFile):
            with open(targetFile, 'rb')as f:
                data = f.read()
            # 读取原图，未处理
            response = make_response(data)
            
            # 设置返回图片的类型
            ext = filename.rsplit('.', 1)[1]
            data_type = 'png'
            if ext == 'jpg' or ext == 'JPG':
                data_type = 'jpg'
            elif ext == 'jpeg' or ext == 'JPEG':
                data_type = 'jpeg'
            else:
                pass
            response.headers['Content-Type'] = 'image/' + data_type
            return response
        else:
            return jsonify({"error": 1001, "msg": "no such file"})
    else:
        pass
    
    
# 
@app.route('/get_gray', methods=['GET'])
def get_processed_img():
    data = base64.b64encode(img)
    return render_template('index1.html',img_stream=data)
 

# list images' names
@app.route('/list_images', methods = ['GET'])
def list_images():
    '''
    返回历史所有上传过的图片的重命名名称
    '''
    if request.method == 'GET':
        file_dir = os.path.join(basedir, app.config['UPLOAD_FOLDER'])
        # base_url = 'http://172.18.167.9:9000/download/'
        base_url = 'http://172.18.167.9:9000/get_image/'
        files = {} 
        urls = [base_url+name for name in os.listdir(file_dir) if allowed_file(name)]
        files['data'] = urls

        return jsonify(files)
    else:
        pass


# 删除图片
@app.route('/delete/<string:filename>', methods=['DELETE'])
def delete_from_server(filename):
    targetFile = os.path.join('upload', filename)
    if request.method == 'DELETE':
        if os.path.isfile(targetFile):
            os.remove(targetFile)
            return jsonify({"success": 0, "msg": "delete successfully"})
        else:
            return jsonify({"error": 1001, "msg": "failed to delete, no such file"})
    else:
        pass


# 显著性分析
@app.route('/process_sa/<string:filename>', methods=['GET'])
def process_sa(filename):
    targetFile = os.path.join('upload', filename)
    if request.method == 'GET':
        data = b""
        if os.path.exists(targetFile):
            with open(targetFile, 'rb')as f:
                data = f.read()
            # 读取原图，未处理
            response = make_response(data)
            
            # 设置返回图片的类型
            ext = filename.rsplit('.', 1)[1]
            data_type = 'png'
            if ext == 'jpg' or ext == 'JPG':
                data_type = 'jpg'
            elif ext == 'jpeg' or ext == 'JPEG':
                data_type = 'jpeg'
            else:
                pass
            response.headers['Content-Type'] = 'image/' + data_type
            return response
        else:
            return jsonify({"error": 1001, "msg": "no such file"})
    else:
        pass
    
    
# 处理图片：背景虚化
@app.route('/process_bokeh/<string:filename>', methods=['GET'])
def process_bokeh(filename):
    targetFile = os.path.join('upload', filename)
    if request.method == 'GET':
        data = b""
        if os.path.exists(targetFile):
            with open(targetFile, 'rb')as f:
                data = f.read()
            # 读取原图，未处理
            response = make_response(data)
            
            # 设置返回图片的类型
            ext = filename.rsplit('.', 1)[1]
            data_type = 'png'
            if ext == 'jpg' or ext == 'JPG':
                data_type = 'jpg'
            elif ext == 'jpeg' or ext == 'JPEG':
                data_type = 'jpeg'
            else:
                pass
            response.headers['Content-Type'] = 'image/' + data_type
            return response
        else:
            return jsonify({"error": 1001, "msg": "no such file"})
    else:
        pass


# 处理图片：抠图
@app.route('/process_cutout/<string:filename>', methods=['GET'])
def process_cutout(filename):
    targetFile = os.path.join('upload', filename)
    if request.method == 'GET':
        data = b""
        if os.path.exists(targetFile):
            with open(targetFile, 'rb')as f:
                data = f.read()
            # 读取原图，未处理
            response = make_response(data)
            
            # 设置返回图片的类型
            ext = filename.rsplit('.', 1)[1]
            data_type = 'png'
            if ext == 'jpg' or ext == 'JPG':
                data_type = 'jpg'
            elif ext == 'jpeg' or ext == 'JPEG':
                data_type = 'jpeg'
            else:
                pass
            response.headers['Content-Type'] = 'image/' + data_type
            return response
        else:
            return jsonify({"error": 1001, "msg": "no such file"})
    else:
        pass


# 处理视频：弹幕防挡
@app.route('/process_barrage/<string:filename>', methods=['GET'])
def process_barrage(filename):
    # to-do https://www.runoob.com/http/http-content-type.html
    targetFile = os.path.join('upload', filename)
    if request.method == 'GET':
        data = b""
        if os.path.exists(targetFile):
            with open(targetFile, 'rb')as f:
                data = f.read()
            # 读取原图，未处理
            response = make_response(data)
            
            # 设置返回图片的类型，mp4待考虑
            ext = filename.rsplit('.', 1)[1]
            data_type = 'png'
            if ext == 'jpg' or ext == 'JPG':
                data_type = 'jpg'
            elif ext == 'jpeg' or ext == 'JPEG':
                data_type = 'jpeg'
            else:
                pass
            response.headers['Content-Type'] = 'image/' + data_type
            return response
        else:
            return jsonify({"error": 1001, "msg": "no such file"})
    else:
        pass
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000)
