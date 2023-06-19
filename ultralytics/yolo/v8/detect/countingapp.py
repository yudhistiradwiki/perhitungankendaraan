# Muhammad Dwiki Yudhistira
# Tugas Akhir
#------------------- Library untuk YOLO ---------------------
import hydra
import torch
import argparse
import time
from pathlib import Path
import cv2
import platform
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from collections import deque
import numpy as np

#---------------------Library untuk Flask WEB ----------------
from flask import Flask, render_template, Response,jsonify,request,session, url_for, redirect
from flask_mysqldb import MySQL
import MySQLdb.cursors, re, hashlib
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField,StringField,DecimalRangeField,IntegerRangeField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired,NumberRange
import os
# import pywhatkit, pyautogui, time

#------------------- Palette warna class --------------------
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)



#---------------- inisialisasi tracking ---------------------
data_deque = {}
deepsort = None

#---------------- Variabel jumlah kendaraan -----------------
object_counter = {}
object_counter1 = {}

#---------- garis counter --------------
line = [(0, 345), (638, 345)] 
# webcam 
# line = [(10, 245), (628, 245)]

#----------- Fungsi Tracking deepsort ------------------
def init_tracker():
    global deepsort
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")

    deepsort= DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                            max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT, nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                            use_cuda=True)

#------------ Fungsi untuk menentukan relativitas kotak ke pixel -----------
def xyxy_to_xywh(*xyxy):
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h

def xyxy_to_tlwh(bbox_xyxy):
    tlwh_bboxs = []
    for i, box in enumerate(bbox_xyxy):
        x1, y1, x2, y2 = [int(i) for i in box]
        top = x1
        left = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        tlwh_obj = [top, left, w, h]
        tlwh_bboxs.append(tlwh_obj)
    return tlwh_bboxs

#---------- fungsi warna kelas --------------
def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    if label == 0: #motor
        color = (85,45,255)
    elif label == 2: # mobil
        color = (222,82,175)
    elif label == 3:  # truk
        color = (0, 204, 255)
    elif label == 5:  # Bus
        color = (0, 149, 255)
    else:
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

#--------- fungsi membuat border dari YOLO -----------
def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1,y1 = pt1
    x2,y2 = pt2
    
    # kiri atas
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)

    # kanan atas
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)

    # kiri bawah
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)

    # kanan bawah
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)
    
    cv2.circle(img, (x1 +r, y1+r), 2, color, 12)
    cv2.circle(img, (x2 -r, y1+r), 2, color, 12)
    cv2.circle(img, (x1 +r, y2-r), 2, color, 12)
    cv2.circle(img, (x2 -r, y2-r), 2, color, 12)
    
    return img

#------------- Penamaan class di border ----------
def UI_box(x, img, color=None, label=None, line_thickness=None):
    # penamaan class di tiap bounding box
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        img = draw_border(img, (c1[0], c1[1] - t_size[1] -3), (c1[0] + t_size[0], c1[1]+3), color, 1, 8, 2)
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

#--------- Fungsi Intersect --------------
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])


#------- Fungsi mendapatkan arah dari sumber kendaraan --------------
def get_direction(point1, point2):
    direction_str = ""

    # arah y axis
    if point1[1] > point2[1]:
        direction_str += "South" # Keluar PEI
    elif point1[1] < point2[1]:
        direction_str += "North" # Masuk PEI
    else:
        direction_str += ""

    # arah x axis
    # if point1[0] > point2[0]:
    #     direction_str += "East"
    # elif point1[0] < point2[0]:
    #     direction_str += "West"
    # else:
    #     direction_str += ""
    return direction_str


# -------------------- Fungsi UI tiap Kelas Kendaraan ---------------
def draw_boxes(img, bbox, names,object_id, identities=None, offset=(0, 0)):
    cv2.line(img, line[0], line[1], (46,162,112), 3)

    height, width, _ = img.shape
    # menghilangkan tracker point jika objek tidak ada
    for key in list(data_deque):
      if key not in identities:
        data_deque.pop(key)

    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        # mencari titik tengah
        center = (int((x2+x1)/ 2), int((y2+y2)/2))

        # get ID dari objek
        id = int(identities[i]) if identities is not None else 0

        # membuat buffer baru untuk objek baru
        if id not in data_deque:  
          data_deque[id] = deque(maxlen= 64)
        color = compute_color_for_labels(object_id[i])
        obj_name = names[object_id[i]]
        label = '{}{:d}'.format("", id) + ":"+ '%s' % (obj_name)

        # titik tengah ke buffer
        data_deque[id].appendleft(center)
        if len(data_deque[id]) >= 2:
          direction = get_direction(data_deque[id][0], data_deque[id][1])
          if intersect(data_deque[id][0], data_deque[id][1], line[0], line[1]):
              cv2.line(img, line[0], line[1], (255, 255, 255), 3)
              if "South" in direction:
                if obj_name not in object_counter:
                    object_counter[obj_name] = 1
                else:
                    object_counter[obj_name] += 1
              if "North" in direction:
                if obj_name not in object_counter1:
                    object_counter1[obj_name] = 1
                else:
                    object_counter1[obj_name] += 1
        
        UI_box(box, img, label=label, color=color, line_thickness=2)
        # gambar garis
        for i in range(1, len(data_deque[id])):
            # cek value none
            if data_deque[id][i - 1] is None or data_deque[id][i] is None:
                continue
            # membuat thickness garis
            thickness = int(np.sqrt(64 / float(i + i)) * 1.5)
            # gambar garis penghitung
            cv2.line(img, data_deque[id][i - 1], data_deque[id][i], color, thickness)
    
    #penghitungan diatas
        for idx, (key, value) in enumerate(object_counter1.items()):
            cnt_str = str(key) + ":" +str(value)
            cv2.line(img, (width - 200,25), (width,25), [85,45,255], 40)
            cv2.putText(img, f'Keluar PEI', (width - 200, 35), 0, 1, [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)
            cv2.line(img, (width - 150, 65 + (idx*40)), (width, 65 + (idx*40)), [85, 45, 255], 30)
            cv2.putText(img, cnt_str, (width - 150, 75 + (idx*40)), 0, 1, [255, 255, 255], thickness = 1, lineType = cv2.LINE_AA)
        for idx, (key, value) in enumerate(object_counter.items()):
            cnt_str1 = str(key) + ":" +str(value)
            cv2.line(img, (11,25), (250,25), [85,45,255], 40)
            cv2.putText(img, f'Masuk PEI', (11, 35), 0, 1, [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)    
            cv2.line(img, (20,65+ (idx*40)), (127,65+ (idx*40)), [85,45,255], 30)
            cv2.putText(img, cnt_str1, (11, 75+ (idx*40)), 0, 1, [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)
    
    return img

#----------- Kelas Predictor dari YOLO ------------------
class DetectionPredictor(BasePredictor):
    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()
        return preds

    def write_results(self, idx, preds, batch):
        p, im, im0 = batch
        all_outputs = []
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        im0 = im0.copy()
        if self.webcam:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)

        self.data_path = p
        save_path = str(self.save_dir / p.name)  # im.jpg
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)

        det = preds[idx]
        all_outputs.append(det)
        if len(det) == 0:
            return log_string
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()  # detections per class
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "
        # write
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        xywh_bboxs = []
        confs = []
        oids = []
        outputs = []
        for *xyxy, conf, cls in reversed(det):
            x_c, y_c, bbox_w, bbox_h = xyxy_to_xywh(*xyxy)
            xywh_obj = [x_c, y_c, bbox_w, bbox_h]
            xywh_bboxs.append(xywh_obj)
            confs.append([conf.item()])
            oids.append(int(cls))
        xywhs = torch.Tensor(xywh_bboxs)
        confss = torch.Tensor(confs)
          
        outputs = deepsort.update(xywhs, confss, oids, im0)
        if len(outputs) > 0:
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -2]
            object_id = outputs[:, -1]
            draw_boxes(im0, bbox_xyxy, self.model.names, object_id,identities)
        return log_string
    
    def show(self, p):
        im0 = self.annotator.result()
        if platform.system() == 'Linux' and p not in self.windows:
            self.windows.append(p)
            cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
            cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
        cv2.imshow(str(p), im0)
        # for detection_ in im0:
        #     ref,buffer=cv2.imencode('.jpg',detection_)
        #     frame=buffer.tobytes()
        #     yield (b'--frame\r\n'
        #             b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')
        cv2.waitKey(1)  # 1 millisecond
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     cv2.destroyAllWindows()
        #     mycursor = mydb.cursor()
        #     columns = ', '.join("`" + str(x).replace('/', '_') + "`" for x in object_counter.keys())
        #     values = ', '.join("'" + str(x).replace('/', '_') + "'" for x in object_counter.values())
        #     sql = "INSERT INTO %s ( %s ) VALUES ( %s );" % ('datacounter', columns, values)
        #     mycursor.execute(sql)
        print(object_counter)
        print(object_counter1)
    
    def count(self):
        return self.object_counter, self.object_counter1

#------------ Fungsi menjalankan class YOLO -----------------
@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    init_tracker()
    cfg.model = "baru.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
    cfg.source = "http://192.168.0.127:4747/video" #"2"
    # cfg.source = "2"
    predictor = DetectionPredictor(cfg)
    predictor()


#------------- app web flask --------------
app = Flask(__name__)
app.config['SECRET_KEY'] = 'muhammaddwiki'
app.config['UPLOAD_FOLDER'] = 'static/files'
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'counter'
mysql = MySQL(app)


#-------------- Login & Logout -----------------
@app.route('/', methods=['GET','POST'])
@app.route('/login', methods=['GET','POST'])
def login():
    msg = ''
    if request.method == 'GET':
        if 'loggedin' in session:
            return redirect('/home')
        return render_template('login.html')

    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        username = request.form['username']
        password = request.form['password']
        # Retrieve the hashed password
        hash = password + app.secret_key
        hash = hashlib.sha1(hash.encode())
        # password = hash.hexdigest()
        # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE username = %s AND password = %s', (username, password,))
        # Fetch one record and return the result
        account = cursor.fetchone()
        if account:
            # Create session data, we can access this data in other routes
            session['loggedin'] = True
            session['id'] = account['id']
            session['username'] = account['username']
            # Redirect to home page
            return redirect('/home')
        else:
            # Account doesnt exist or username/password incorrect
            msg = "maaf salah"
    return render_template('login.html', msg=msg)

@app.route('/logout')
def logout():
     # Remove session data, this will log the user out
   session.pop('loggedin', None)
   session.pop('id', None)
   session.pop('username', None)
   # Redirect to login page
   return redirect(url_for('login'))


#-------------- Dashboard ----------------
@app.route('/home')
def home():
    if request.method == 'GET' and 'loggedin' in session:
        cursor = mysql.connection.cursor()
        cursor.execute(''' SELECT ROUND(AVG(motor)) FROM datacounter ''')
        motor = cursor.fetchone()
        cursor.execute(''' SELECT ROUND(AVG(mobil)) FROM datacounter ''')
        mobil = cursor.fetchone()
        cursor.execute(''' SELECT ROUND(AVG(truk)) FROM datacounter ''')
        truk = cursor.fetchone()
        cursor.execute(''' SELECT ROUND(AVG(bus)) FROM datacounter ''')
        bus = cursor.fetchone()
        cursor.execute(''' SELECT tanggal FROM datacounter ''')
        tanggal = cursor.fetchall()
        return render_template('index.html', motor=motor, mobil=mobil, truk=truk, bus=bus, tanggal=tanggal)
    return redirect(url_for('login'))

#---------------- route menu hitung ----------------
@app.route("/hitung", methods=['GET','POST'])
def hitung():
    if request.method == 'GET' and 'loggedin' in session:
        return render_template('perhitungan.html')
    return redirect(url_for('login'))


#----------------- menampilkan kamera ---------------
@app.route('/webapp')
def webapp():
    #return Response(generate_frames(path_x = session.get('video_path', None),conf_=round(float(session.get('conf_', None))/100,2)),mimetype='multipart/x-mixed-replace; boundary=frame')
    # return Response(predict(), mimetype='multipart/x-mixed-replace; boundary=frame')
    predict()
    cursor = mysql.connection.cursor()
    columns = ', '.join("`" + str(x).replace('/', '_') + "`" for x in object_counter.keys())
    values = ', '.join("'" + str(x).replace('/', '_') + "'" for x in object_counter.values())
    sql = "INSERT INTO %s (%s) VALUES (%s);" % ('datacounter', columns, values)
    cursor.execute(sql)
    mysql.connection.commit()
    telegram = "Kirim via whatsapp"
    return render_template('perhitungan.html', keluar=str(object_counter1), masuk=str(object_counter), telegram=telegram)

@app.route('/send', methods=['GET','POST'])
def send():
    import pywhatkit, pyautogui, time, datetime
    if request.method == 'POST':
        masuk = request.form['masuk']
        keluar = request.form['keluar']
        pywhatkit.sendwhatmsg("+6289627286733", "Kendaraan Masuk : " + str(masuk) + "Kendaraan Keluar : " + str(keluar), int(datetime.datetime.now().hour), int(datetime.datetime.now().minute) + 2 )
        time.sleep(1)
        pyautogui.click()
        time.sleep(1)
        pyautogui.press('enter')
        telegram = "Kirim via whatsapp"
    return render_template('perhitungan.html', keluar=keluar, masuk=masuk, telegram=telegram)
    

if __name__ == "__main__":
    app.run(debug=True)