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
    
    result_dir = os.path.join(basedir, app.config['SA_RESULT_FOLDER'])
    if not os.path.exists(result_dir):
            os.makedirs(result_dir)
            
    if request.method == 'GET':
        # 定义device
        device = torch.device('cuda:0')
        
        data = b""
        if os.path.exists(targetFile):
            
#             # 把目标文件读成pil.image形式
#             ori_img = Image.open(targetFile)
#             # 获取sa
#             sa_result = createSOD(ori_img, model, device)
#             # 保存？是否需要??
#             save_file_path = os.path.join(result_dir, filename)
#             sa_result.save(save_file_path)
            
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
    
    result_dir = os.path.join(basedir, app.config['BOKEH_RESULT_FOLDER'])
    if not os.path.exists(result_dir):
            os.makedirs(result_dir)
            
    if request.method == 'GET':
        data = b""
        if os.path.exists(targetFile):
            
#             # 把目标文件读成pil.image形式
#             ori_img = Image.open(targetFile)
#             # 获取sa
#             sa_result = createSOD(ori_img, model, device)
#             # 获取bokeh
#             bokeh_result = background_blur(ori_img, sa_result)
#             # 保存？是否需要??
#             save_file_path = os.path.join(result_dir, filename)
#             bokeh_result.save(save_file_path)
            
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
    
    sodForTargetFile = os.path.join('result/sa', filename)
    
    result_dir = os.path.join(basedir, app.config['CUTOUT_RESULT_FOLDER'])
    if not os.path.exists(result_dir):
            os.makedirs(result_dir)
    
    if request.method == 'GET':
        data = b""
        if os.path.exists(targetFile):
#             # 把目标文件读成pil.image形式
#             ori_img = Image.open(targetFile)
#             # 获取sa
#             sa_result = createSOD(ori_img, model, device)
#             # 获取cutout
#             cutout_result = createMaskForPicture(ori_img, sa_result)
#             # 保存为png？是否需要
#             save_file_path = os.path.join(result_dir, filename)
#             cutout_result.save(save_file_path)
            
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
    
    result_dir = os.path.join(basedir, app.config['BARRAGE_RESULT_FOLDER'])
    if not os.path.exists(result_dir):
            os.makedirs(result_dir)
    
    if request.method == 'GET':
        # 定义device
        device = torch.device('cuda:3')
        
        data = b""
        if os.path.exists(targetFile):
#             # 把目标文件读成pil.image形式
#             ori_img = Image.open(targetFile)
#             # 获取barrage
#             sa_result = createSOD(ori_img, model, device)
#             # 获取背景图？？
#             bg_img = # ??? 
#             #获取barrage
#             barrage_result = background_substitution(ori_img, sa_result, bg_img)
#             # 保存为png？是否需要
#             save_file_path = os.path.join(result_dir, filename)
#             barrage_result.save(save_file_path)
            
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
    
    rootDir = './imgs/'
    resultDir = './SSRN_DUTS_v2/'
    img1 = Image.open(os.path.join(rootDir, "2092.jpg"))
    img2 = Image.open(os.path.join(resultDir, "2092.png"))
    plt.imshow(img1)

    #使用模型
    if False:
        sys.path.append("../..")
        from saliency_pytorch.ssrnet import SSRNet
        class ImageModel(nn.Module):
            def __init__(self, pretrained = False):
                super(ImageModel, self).__init__()
                self.backbone = SSRNet(1, 16, pretrained=pretrained)

            def forward(self, frame):
                seg = self.backbone(frame)
                return seg
        # device = torch.device('cuda:1')
        model = ImageModel(pretrained=True)
        model_path = "../../saliency_pytorch/image_miou_087.pth"
        model.load_state_dict(torch.load(model_path), strict = True)
