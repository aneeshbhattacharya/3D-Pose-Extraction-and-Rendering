from argparse import ArgumentParser
import json
import os

import cv2
import numpy as np
import glob

from modules.input_reader import VideoReader, ImageReader
from modules.draw import Plotter3d, draw_poses
from modules.parse_poses import parse_poses


def rotate_poses(poses_3d, R, t):
    R_inv = np.linalg.inv(R)
    for pose_id in range(len(poses_3d)):
        pose_3d = poses_3d[pose_id].reshape((-1, 4)).transpose()
        pose_3d[0:3, :] = np.dot(R_inv, pose_3d[0:3, :] - t)
        poses_3d[pose_id] = pose_3d.transpose().reshape(-1)

    return poses_3d

parser = ArgumentParser(description='Lightweight 3D human pose estimation demo. '
                                    'Press esc to exit, "p" to (un)pause video or process next image.')
parser.add_argument('-m', '--model',
                    help='Required. Path to checkpoint with a trained model '
                            '(or an .xml file in case of OpenVINO inference).',
                    type=str, required=True)
parser.add_argument('--video', help='Optional. Path to video file or camera id.', type=str, required=True)
parser.add_argument('-d', '--device',
                    help='Optional. Specify the target device to infer on: CPU or GPU. '
                            'The demo will look for a suitable plugin for device specified '
                            '(by default, it is GPU).',
                    type=str, default='GPU')
parser.add_argument('--use-openvino',
                    help='Optional. Run network with OpenVINO as inference engine. '
                            'CPU, GPU, FPGA, HDDL or MYRIAD devices are supported.',
                    action='store_true')
parser.add_argument('--use-tensorrt', help='Optional. Run network with TensorRT as inference engine.',
                    action='store_true')
parser.add_argument('--images', help='Optional. Path to input image(s).', nargs='+', default='')
parser.add_argument('--height-size', help='Optional. Network input layer height size.', type=int, default=256)
parser.add_argument('--extrinsics-path',
                    help='Optional. Path to file with camera extrinsics.',
                    type=str, default=None)
parser.add_argument('--fx', type=np.float32, default=-1, help='Optional. Camera focal length.')

args = parser.parse_args()

video_path = args.video
save_directory = './POSES'

if os.path.exists(save_directory) == False:
    os.makedirs(save_directory)
    os.makedirs(save_directory+"/poses")
    os.makedirs(save_directory+"/info")

video_id = os.path.basename(video_path)

if args.video == '' and args.images == '':
    raise ValueError('Either --video or --image has to be provided')

numpy_save_name = video_id

stride = 8
if args.use_openvino:
    from modules.inference_engine_openvino import InferenceEngineOpenVINO
    net = InferenceEngineOpenVINO(args.model, args.device)
else:
    from modules.inference_engine_pytorch import InferenceEnginePyTorch
    net = InferenceEnginePyTorch(args.model, args.device, use_tensorrt=args.use_tensorrt)

file_path = args.extrinsics_path
if file_path is None:
    file_path = os.path.join('data', 'extrinsics.json')
with open(file_path, 'r') as f:
    extrinsics = json.load(f)
R = np.array(extrinsics['R'], dtype=np.float32)
t = np.array(extrinsics['t'], dtype=np.float32)

frame_provider = ImageReader(args.images)
is_video = False
if args.video != '':
    frame_provider = VideoReader(args.video)
    is_video = True
base_height = args.height_size
fx = args.fx

delay = 1
esc_code = 27
p_code = 112
space_code = 32
mean_time = 0

#Final 3D pose list

pose_3d_FINAL = []
frame_count = 0

for frame in frame_provider:
    frame_count+=1
    current_time = cv2.getTickCount()
    if frame is None:
        print("ENDED LOOP")
        break
    input_scale = base_height / frame.shape[0]
    scaled_img = cv2.resize(frame, dsize=None, fx=input_scale, fy=input_scale)
    scaled_img = scaled_img[:, 0:scaled_img.shape[1] - (scaled_img.shape[1] % stride)]  # better to pad, but cut out for demo
    if fx < 0:  # Focal length is unknown
        fx = np.float32(0.8 * frame.shape[1])

    inference_result = net.infer(scaled_img)
    poses_3d, poses_2d = parse_poses(inference_result, input_scale, stride, fx, is_video)
    

    edges = []
    if len(poses_3d):
        poses_3d = rotate_poses(poses_3d, R, t)
        poses_3d_copy = poses_3d.copy()
        x = poses_3d_copy[:, 0::4]
        y = poses_3d_copy[:, 1::4]
        z = poses_3d_copy[:, 2::4]
        poses_3d[:, 0::4], poses_3d[:, 1::4], poses_3d[:, 2::4] = -z, x, -y

        poses_3d = poses_3d.reshape(poses_3d.shape[0], 19, -1)[:, :, 0:3]

        # KEEP SAVING 3D poses:
        # print(poses_3d)

        part1 = poses_3d[:2]
        part2 = poses_3d[3:]
        poses_3d = np.vstack((part1,part2))


        pose_3d_FINAL.append(poses_3d)

pose_3d_FINAL = np.array(pose_3d_FINAL)


with open(save_directory+'/poses/'+numpy_save_name+'_poses.npy','wb') as f:
    np.save(f,pose_3d_FINAL)

with open(save_directory+'/info/'+numpy_save_name+"_frames.txt",'w') as f:
    f.write(str(frame_count))

print("FRAME COUNT: ",frame_count)

