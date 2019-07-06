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
from PIL import Image
from urllib.request import urlretrieve
from utils import *  # 调用network有关函数
import torch
import torch.nn as nn

app = Flask(__name__)

# 解决跨域访问问题
# r'/*' 是通配符，让本服务器所有的URL 都允许跨域请求
CORS(app, supports_credentials=True, resources=r'/*')

UPLOAD_FOLDER = 'upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

BG_FOLDER = 'background'
app.config['BG_FOLDER'] = BG_FOLDER

SA_RESULT_FOLDER = 'result/sa'
app.config['SA_RESULT_FOLDER'] = SA_RESULT_FOLDER

BOKEH_RESULT_FOLDER = 'result/bokeh'
app.config['BOKEH_RESULT_FOLDER'] = BOKEH_RESULT_FOLDER

CUTOUT_RESULT_FOLDER = 'result/cutout'
app.config['CUTOUT_RESULT_FOLDER'] = CUTOUT_RESULT_FOLDER

BARRAGE_RESULT_FOLDER = 'result/barrage'
app.config['BARRAGE_RESULT_FOLDER'] = BARRAGE_RESULT_FOLDER

basedir = os.path.dirname(os.path.abspath("__file__"))
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'gif', 'GIF', 'mp4', 'jpeg', 'JPEG'])

# 使用模型
sys.path.append("../..")
# device = torch.device('cuda:1')
model = ImageModel(pretrained=True)
model_path = "./saliency_pytorch/image_miou_087.pth"
model.load_state_dict(torch.load(model_path), strict = True)


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
            
            ext = fname.rsplit('.', 1)[1]
            # !!! 返回new_filename，使得前端可以直接访问被重命名的图片
            new_filename = Pic_str().create_uuid() + '.' + ext
            
            if allowed_file(new_filename):
                urlretrieve(img_url, os.path.join(file_dir, new_filename))
                print(fname, 'upload success')
                # 成功，返回保存的文件名，以便前端可利用
                return jsonify({"success": 0, "msg": "upload successfully", "name": new_filename})
            else:
                return jsonify({"error": 1001, "msg": "failed to upload", "name": fname})
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
                return jsonify({"error": 1001, "msg": "failed to upload", "name": fname})
    else:
        pass
    
    
# 上传背景图片
@app.route('/upload_bg', methods = ['POST'])
def upload_bg():
    file_dir = os.path.join(basedir, app.config['BG_FOLDER'])
    if request.method == 'POST':
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        if request.data != b'':
            url_data = str(request.data, encoding='utf-8')
            img_url = url_data[8:-2]
            img_url = str(img_url)
            url_split = img_url.split('/')
            fname = url_split[-1]
            
            ext = fname.rsplit('.', 1)[1]
            # !!! 返回new_filename，使得前端可以直接访问被重命名的图片
            new_filename = Pic_str().create_uuid() + '.' + ext
            
            if allowed_file(new_filename):
                urlretrieve(img_url, os.path.join(file_dir, new_filename))
                print(fname, 'upload success')
                # 成功，返回保存的文件名，以便前端可利用
                return jsonify({"success": 0, "msg": "upload successfully", "name": new_filename})
            else:
                return jsonify({"error": 1001, "msg": "failed to upload", "name": fname})
        else:
            f = request.files['file']
            if f and allowed_file(f.filename):
                fname = secure_filename(f.filename)

                # 把图片保存在upload文件目录下
                f.save(os.path.join(file_dir, fname))
                print(fname, 'upload success')
                # 成功，返回保存的文件名，以便前端可利用
                return jsonify({"success": 0, "msg": "upload bg successfully", "name": fname})
            else:
                return jsonify({"error": 1001, "msg": "failed to upload", "name": fname})
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
    

# list images' names
@app.route('/list_images', methods = ['GET'])
def list_images():
    '''
    返回历史所有上传过的图片的名称
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
    
    
# 获取背景图
@app.route('/get_background/<string:filename>', methods=['GET'])
def get_background(filename):
    global bgFilename
    targetFile = os.path.join('background', filename)
    bgFilename = targetFile
    
    if request.method == 'GET':
        data = b""
        if os.path.exists(targetFile):
            with open(targetFile, 'rb')as f:
                data = f.read()
            # 读取背景图
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
        pass
    
 
# list backgrounds url
@app.route('/list_background', methods = ['GET'])
def list_background():
    '''
    返回历史所有上传过的图片的名称
    '''
    if request.method == 'GET':
        file_dir = os.path.join(basedir, app.config['BG_FOLDER'])
        # base_url = 'http://172.18.167.9:9000/download/'
        base_url = 'http://172.18.167.9:9000/get_background/'
        files = {} 
        urls = [base_url+name for name in os.listdir(file_dir) if allowed_file(name)]
        files['data'] = urls

        return jsonify(files)
    else:
        pass
    
    
# 获取视频帧文件夹
@app.route('/get_video_frame/<string:filename>', methods=['GET'])
def get_video_frame(filename):
    
    filepath = os.path.expanduser("/data3/yanpengxiang/datasets/DAVIS2016/JPEGImages/480p")
   
    tmp_folder, tmp_filename = filename.split('_')[0], filename.split('_')[1]
    targetFile = os.path.join(filepath, tmp_folder, tmp_filename)
    
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
    
    
# 获取视频帧文件夹
@app.route('/get_video_mask/<string:filename>', methods=['GET'])
def get_video_mask(filename):
    
    filepath = os.path.expanduser("/data3/yanpengxiang/datasets/DAVIS2016/results")
   
    tmp_folder, tmp_filename = filename.split('_')[0], filename.split('_')[1]
    targetFile = os.path.join(filepath, tmp_folder, tmp_filename)
    
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

    
# 删除背景图片
@app.route('/delete_bg/<string:filename>', methods=['DELETE'])
def delete_bg_from_server(filename):
    targetFile = os.path.join('background', filename)
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
    
#     result_dir = os.path.join(basedir, app.config['SA_RESULT_FOLDER'])
#     if not os.path.exists(result_dir):
#             os.makedirs(result_dir)
            
    if request.method == 'GET':
        # 定义device
        device = torch.device('cuda:0')
        
        data = b""
        if os.path.exists(targetFile):
            # 设置返回图片的类型
            ext = filename.rsplit('.', 1)[1]
            data_type = 'png'
            if ext == 'jpg' or ext == 'JPG':
                data_type = 'jpg'
            elif ext == 'jpeg' or ext == 'JPEG':
                data_type = 'jpeg'
            else:
                pass
            
            # 把目标文件读成pil.image形式
            ori_img = Image.open(targetFile)
            
            # 获取sa
            sa_result = createSOD(ori_img, model, device)
            data = cv2.imencode('.' + data_type, sa_result)[1].tobytes()
#             # 保存？是否需要??
#             save_file_path = os.path.join(result_dir, filename)
#             sa_result.save(save_file_path)
            
#             with open(save_file_path, 'rb')as f:
#                 data = f.read()
            # 读取原图，未处理
            response = make_response(data)
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
        device = torch.device('cuda:0')
        data = b""
        if os.path.exists(targetFile):
            
            # 把目标文件读成pil.image形式
            ori_img = Image.open(targetFile)
            # 获取sa
            sa_result = createSOD(ori_img, model, device)
            sa_result = Image.fromarray(np.uint8(sa_result))
            # 获取bokeh
            bokeh_result = background_blur(ori_img, sa_result, [100], [13])
            
            # 设置返回图片的类型
            ext = filename.rsplit('.', 1)[1]
            data_type = 'png'
            if ext == 'jpg' or ext == 'JPG':
                data_type = 'jpg'
            elif ext == 'jpeg' or ext == 'JPEG':
                data_type = 'jpeg'
            else:
                pass
            
            data = cv2.imencode('.' + data_type, bokeh_result)[1].tobytes()

            response = make_response(data)

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
        device = torch.device('cuda:1')
        data = b""
        if os.path.exists(targetFile):
            # 把目标文件读成pil.image形式
            ori_img = Image.open(targetFile)
            # 获取sa
            sa_result = createSOD(ori_img, model, device)
            sa_result = Image.fromarray(np.uint8(sa_result))
            # 获取cutout
            cutout_result = createMaskForPicture(ori_img, sa_result)
            
            # 设置返回图片的类型
            ext = filename.rsplit('.', 1)[1]
            data_type = 'png'
            if ext == 'jpg' or ext == 'JPG':
                data_type = 'jpg'
            elif ext == 'jpeg' or ext == 'JPEG':
                data_type = 'jpeg'
            else:
                pass
            
            data = cv2.imencode('.png' , cutout_result)[1].tobytes()

            response = make_response(data)
        
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
        # 定义device
        device = torch.device('cuda:2')
        
        data = b""
        if os.path.exists(targetFile):
            # 把目标文件读成pil.image形式
            ori_img = Image.open(targetFile)
            # 获取barrage
            sa_result = createSOD(ori_img, model, device)
            sa_result = Image.fromarray(np.uint8(sa_result))
            #获取barrage
            barrage_result = barrage(ori_img, sa_result)
            
            # 设置返回图片的类型，mp4待考虑
            ext = filename.rsplit('.', 1)[1]
            data_type = 'png'
            if ext == 'jpg' or ext == 'JPG':
                data_type = 'jpg'
            elif ext == 'jpeg' or ext == 'JPEG':
                data_type = 'jpeg'
            else:
                pass
            
            data = cv2.imencode('.' + data_type, barrage_result)[1].tobytes()
            
            response = make_response(data)

            response.headers['Content-Type'] = 'image/' + data_type
            return response
        else:
            return jsonify({"error": 1001, "msg": "no such file"})
    else:
        pass


# 背景替换
@app.route('/process_replace/<string:filename>', methods=['GET'])
def process_replacement(filename):
    targetFile = os.path.join('upload', filename)

    if request.method == 'GET':
        # 定义device
        device = torch.device('cuda:1')
        
        data = b""
        if os.path.exists(targetFile):
            # 把目标文件读成pil.image形式
            ori_img = Image.open(targetFile)
            # 获取barrage
            sa_result = createSOD(ori_img, model, device)
            sa_result = Image.fromarray(np.uint8(sa_result))
            # 获取背景图
            bg_img = Image.open(bgFilename)
            #获取barrage
            bg_replacement_result = background_substitution(ori_img, sa_result, bg_img, maxThreshold = 180, minThreshold = 80)
            
            # 设置返回图片的类型，mp4待考虑
            ext = filename.rsplit('.', 1)[1]
            data_type = 'png'
            if ext == 'jpg' or ext == 'JPG':
                data_type = 'jpg'
            elif ext == 'jpeg' or ext == 'JPEG':
                data_type = 'jpeg'
            else:
                pass
            
            data = cv2.imencode('.' + data_type, bg_replacement_result)[1].tobytes()

            response = make_response(data)

            response.headers['Content-Type'] = 'image/' + data_type
            return response
        else:
            return jsonify({"error": 1001, "msg": "no such file"})
    else:
        pass
    
# splash of color
@app.route('/process_splash/<string:filename>', methods=['GET'])
def process_splash(filename):
    targetFile = os.path.join('upload', filename)
            
    if request.method == 'GET':
        device = torch.device('cuda:1')
        data = b""
        if os.path.exists(targetFile):
            
            # 把目标文件读成pil.image形式
            ori_img = Image.open(targetFile)
            # 获取sa
            sa_result = createSOD(ori_img, model, device)
            sa_result = Image.fromarray(np.uint8(sa_result))
            # 获取bokeh
            splash_result = createGrayPicture(ori_img, sa_result)
            
            # 设置返回图片的类型
            ext = filename.rsplit('.', 1)[1]
            data_type = 'png'
            if ext == 'jpg' or ext == 'JPG':
                data_type = 'jpg'
            elif ext == 'jpeg' or ext == 'JPEG':
                data_type = 'jpeg'
            else:
                pass
            
            data = cv2.imencode('.' + data_type, splash_result)[1].tobytes()

            response = make_response(data)

            response.headers['Content-Type'] = 'image/' + data_type
            return response
        else:
            return jsonify({"error": 1001, "msg": "no such file"})
    else:
        pass
    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000)
