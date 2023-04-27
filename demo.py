import argparse
import numpy as np
import cv2
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision

from PIL import Image
from PIL import Image, ImageOps
from utils import select_device

from face_detection import RetinaFace
from model import L2CS

from acesvision.ObjectDetection import YOLOX
from acesvision.FaceAlignment2D import FaceAlignment2DAPI, Fan2D


def put_text_with_font(img, text, position, font_path, font_size, color):
    # OpenCVの画像をPIL画像に変換
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # フォントを読み込む
    font = ImageFont.truetype(font_path, font_size)

    # ImageDrawオブジェクトを作成
    draw = ImageDraw.Draw(img_pil)

    # テキストを描画
    draw.text(position, text, font=font, fill=color)

    # PIL画像をOpenCV画像に変換
    img_with_text = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    return img_with_text

def draw_gaze(org_pos,image_in, pitchyaw, thickness=2, color=(255, 255, 0),sclae=2.0):
    """Draw gaze angle on given image with a given eye positions."""
    image_out = image_in
    (h, w) = image_in.shape[:2]
    length = w/4
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    pitch = pitchyaw[0]
    yaw = pitchyaw[1]
    dx = -length * np.sin(pitch) * np.cos(yaw)
    dy = -length * np.sin(yaw)
    cv2.arrowedLine(image_out, tuple(np.round(org_pos).astype(np.int32)),
                   tuple(np.round([org_pos[0] + dx, org_pos[1] + dy]).astype(int)), color,
                   thickness, cv2.LINE_AA, tipLength=0.18)
    return image_out

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Gaze evalution using model pretrained with L2CS-Net on Gaze360.')
    parser.add_argument(
        '--gpu',dest='gpu_id', help='GPU device id to use [0]',
        default="0", type=str)
    parser.add_argument(
        '--snapshot',dest='snapshot', help='Path of model snapshot.',
        default='models/L2CSNet_gaze360.pkl', type=str)
    parser.add_argument(
        '--cam',dest='cam_id', help='Camera device id to use [0]',
        default="data_demo/cell_phone.mp4", type=str)
    parser.add_argument(
        '--arch',dest='arch',help='Network architecture, can be: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152',
        default='ResNet50', type=str)
    parser.add_argument(
        '--save_root', default='outputs', type=Path
    )

    args = parser.parse_args()
    return args

def getArch(arch,bins):
    # Base network structure
    if arch == 'ResNet18':
        model = L2CS( torchvision.models.resnet.BasicBlock,[2, 2,  2, 2], bins)
    elif arch == 'ResNet34':
        model = L2CS( torchvision.models.resnet.BasicBlock,[3, 4,  6, 3], bins)
    elif arch == 'ResNet101':
        model = L2CS( torchvision.models.resnet.Bottleneck,[3, 4, 23, 3], bins)
    elif arch == 'ResNet152':
        model = L2CS( torchvision.models.resnet.Bottleneck,[3, 8, 36, 3], bins)
    else:
        if arch != 'ResNet50':
            print('Invalid value for architecture is passed! '
                'The default value of ResNet50 will be used instead!')
        model = L2CS( torchvision.models.resnet.Bottleneck, [3, 4, 6,  3], bins)
    return model

def extract_result(result, target_label):
    new_ret = []
    for ret in result:
        if ret['label'] in target_label:
            ret['class'] = 0
            new_ret.append(ret)
    return new_ret


def rounded_rectangle(draw, xy, corner_radius, fill=None, outline=None, width=1):
    upper_left_point = xy[0]
    bottom_right_point = xy[1]
    draw.rectangle(
        [
            (upper_left_point[0] + corner_radius, upper_left_point[1]),
            (bottom_right_point[0] - corner_radius, bottom_right_point[1])
        ],
        fill=fill, outline=outline, width=width
    )
    draw.rectangle(
        [
            (upper_left_point[0], upper_left_point[1] + corner_radius),
            (bottom_right_point[0], bottom_right_point[1] - corner_radius)
        ],
        fill=fill, outline=outline, width=width
    )
    draw.ellipse(
        [
            (upper_left_point[0], upper_left_point[1]),
            (upper_left_point[0] + corner_radius * 2, upper_left_point[1] + corner_radius * 2)
        ],
        fill=fill, outline=outline
    )
    draw.ellipse(
        [
            (bottom_right_point[0] - corner_radius * 2, upper_left_point[1]),
            (bottom_right_point[0], upper_left_point[1] + corner_radius * 2)
        ],
        fill=fill, outline=outline
    )
    draw.ellipse(
        [
            (upper_left_point[0], bottom_right_point[1] - corner_radius * 2),
            (upper_left_point[0] + corner_radius * 2, bottom_right_point[1])
        ],
        fill=fill, outline=outline
    )
    draw.ellipse(
        [
            (bottom_right_point[0] - corner_radius * 2, bottom_right_point[1] - corner_radius * 2),
            (bottom_right_point[0], bottom_right_point[1])
        ],
        fill=fill, outline=outline
    )

def put_text_with_background(img, text, org, font_path, font_size, text_color, bg_color, rectangle_color, thickness=1, padding_x=2, padding_y=2, corner_radius=10):
    # OpenCV画像をPIL画像に変換
    # img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img_pil = Image.fromarray(img)

    draw = ImageDraw.Draw(img_pil)

    # フォントをロード
    font = ImageFont.truetype(font_path, font_size)

    # テキストサイズを取得
    text_width, text_height = draw.textsize(text, font=font)

    # テキストの背景領域を定義
    x, y = org
    bg_rect = x - padding_x, y - padding_y, x + text_width + padding_x, y + text_height + padding_y

    # 背景領域に色を塗る
    # draw.rectangle([bg_rect[:2], bg_rect[2:]], fill=bg_color, outline=rectangle_color, width=4)
    rounded_rectangle(draw, [bg_rect[:2], bg_rect[2:]], corner_radius, fill=bg_color, outline=None, width=4)

    # テキストを描画
    draw.text(org, text, font=font, fill=text_color)

    # PIL画像をOpenCV画像に変換して返す
    # return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return np.array(img_pil)


def draw_alert(frame):
    text = "Using cell phone"
    org = (780, 100)
    font = cv2.FONT_HERSHEY_TRIPLEX
    font_scale = 45
    text_color = (255, 255, 255)
    bg_color = (90, 73, 246)

    rect_angle_color = (77, 77, 222)
    thickness = 1
    padding_x = 50
    padding_y = 30
    # def put_text_with_background(img, text, org, font, font_scale, text_color, bg_color, thickness=1, padding=2):
    # put_text_with_background(img=frame, text=text, org=org, font=font, font_scale=font_scale, text_color=text_color, bg_color=bg_color, rectangle_color=rect_angle_color, thickness=thickness, padding=padding)
    frame = put_text_with_background(img=frame, text=text, org=org, font_path='Helvetica_Bold.TTF', font_size=font_scale, text_color=text_color, bg_color=bg_color, rectangle_color=rect_angle_color, thickness=thickness, padding_x=padding_x, padding_y=padding_y)
    return frame


def main():
    args = parse_args()

    cudnn.enabled = True
    arch=args.arch
    batch_size = 1
    cam = args.cam_id
    gpu = select_device(args.gpu_id, batch_size=batch_size)
    snapshot_path = args.snapshot

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    save_path = Path(args.save_root, Path(args.cam_id).stem).with_suffix('.mp4')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    video_writer = None

    transformations = transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    model=getArch(arch, 90)
    print('Loading snapshot.')
    saved_state_dict = torch.load(snapshot_path)
    model.load_state_dict(saved_state_dict)
    model.cuda(gpu)
    model.eval()


    softmax = nn.Softmax(dim=1)
    detector = RetinaFace(gpu_id=0)
    # det_model = YOLOX(model_hash='YOLOX_MaskedLfwWiderface_c1296834')
    pred_model = Fan2D()
    face_det_model = YOLOX(model_hash='YOLOX_MaskedLfwWiderface_c1296834')
    keypoints_predictor = FaceAlignment2DAPI(
        pred_model=pred_model,
        det_model=face_det_model,
        num_max_det=None,
        cropped_size=(256, 256))
    # face_detector = YOLOX(model_hash='YOLOX_MaskedLfwWiderface_c1296834')
    det_model = YOLOX(model_hash='YOLOX_COCO_f68199a3')
    conf_thre = 0.0175
    idx_tensor = [idx for idx in range(90)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)
    x=0

    similarity_threshold = 0.9
    COLORS = [[243, 207, 153], [255, 90, 0], [0, 170, 246], [122, 175, 3], [255, 90, 0], [255, 196, 77], [0, 241, 255], [0, 75, 255]]
    LABEL_FONT = cv2.FONT_HERSHEY_TRIPLEX
    text_bbox_color = None
    text_bbox_thickness = -1
    draw_text_bbox = True
    text_position = "top"
    TEXT_SCALE = None
    TEXT_THICKNESS = 2
    # TEXT_KEYS = ['label', 'bbox_confidence']
    TEXT_KEYS = ['label']
    TEXT_COLOR = [0, 0, 0]
    NMS_THRE = 0.1

    cap = cv2.VideoCapture(cam)

    # default_gaze_color = [222, 243, 202]
    default_gaze_color = [0, 255, 0]
    # alert_gaze_color = [36, 31, 182]
    alert_gaze_color = [0, 0, 255]

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    with torch.no_grad():
        while True:
            success, frame = cap.read()
            if not success:
                break
            start_fps = time.time()
            if video_writer is None:
                height, width = frame.shape[:2]
                fps = round(cap.get(cv2.CAP_PROP_FPS))
                video_writer = cv2.VideoWriter(str(save_path), fourcc, fps, (width, height))

            faces = detector(frame)

            keypoints_results = keypoints_predictor(frame)
            landmarks = np.array(keypoints_results[0][0]['2d_landmarks'])
            r_origin = tuple(landmarks[36:42].mean(0).astype(np.int64))  # origin coordinate of the right eye
            l_origin = tuple(landmarks[42:48].mean(0).astype(np.int64))  # origin coordinate of the left eye
            eye_center = ((r_origin[0] + l_origin[0]) // 2, (r_origin[1] + l_origin[1]) // 2)
            result = det_model(frame[:, :, ::-1], return_name=True, conf_thre=conf_thre, nms_thre=NMS_THRE)[0]
            cellphone_result = extract_result(result, ['cell phone'])
            if faces is not None:
                for box, landmarks, score in faces:
                    if score < .95:
                        continue
                    x_min=int(box[0])
                    if x_min < 0:
                        x_min = 0
                    y_min=int(box[1])
                    if y_min < 0:
                        y_min = 0
                    x_max=int(box[2])
                    y_max=int(box[3])
                    bbox_width = x_max - x_min
                    bbox_height = y_max - y_min

                    # Crop image
                    img = frame[y_min:y_max, x_min:x_max]
                    img = cv2.resize(img, (224, 224))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    im_pil = Image.fromarray(img)
                    img=transformations(im_pil)
                    img  = Variable(img).cuda(gpu)
                    img  = img.unsqueeze(0)

                    # gaze prediction
                    gaze_pitch, gaze_yaw = model(img)

                    pitch_predicted = softmax(gaze_pitch)
                    yaw_predicted = softmax(gaze_yaw)

                    # Get continuous predictions in degrees.
                    pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 4 - 180
                    yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 4 - 180

                    pitch_predicted= pitch_predicted.cpu().detach().numpy()* np.pi/180.0
                    yaw_predicted= yaw_predicted.cpu().detach().numpy()* np.pi/180.0

                    (h, w) = frame.shape[:2]
                    length = w / 2
                    gaze_vector = np.array([-np.sin(pitch_predicted) * np.cos(yaw_predicted), - np.sin(yaw_predicted)])

                    face_center = [(x_min + x_max) / 2, (y_min + y_max) / 2]
                    frame = det_model.draw(
                        frame,
                        cellphone_result,
                        draw_text_bbox=draw_text_bbox,
                        text_keys = TEXT_KEYS,
                        text_font=LABEL_FONT,
                        text_bbox_thickness=text_bbox_thickness,
                        text_bbox_color=text_bbox_color,
                        colors=COLORS,
                        text_color=TEXT_COLOR,
                        label_position=text_position,
                        text_scale=TEXT_SCALE,
                        text_thickness=TEXT_THICKNESS)

                    warning = False
                    for cell_phone in cellphone_result:
                        cell_phone_center = [int(cell_phone['bbox']['x'] + cell_phone['bbox']['w']/ 2), int(cell_phone['bbox']['y'] + cell_phone['bbox']['h'] / 2)]
                        face_cellphone_vector = np.array(cell_phone_center) - np.array(eye_center)

                        similarity = np.dot(gaze_vector, face_cellphone_vector) / (np.linalg.norm(gaze_vector) * np.linalg.norm(face_cellphone_vector))

                        if similarity > similarity_threshold:
                            frame = draw_alert(frame)
                            warning = True

                    if warning:
                        draw_gaze((eye_center),frame,(pitch_predicted,yaw_predicted),color=alert_gaze_color)
                    else:
                        draw_gaze((eye_center),frame,(pitch_predicted,yaw_predicted),color=default_gaze_color)

            video_writer.write(frame)
    video_writer.release()


if __name__ == '__main__':
    main()
