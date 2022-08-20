from argparse import ArgumentParser
import json
import os

import cv2
import numpy as np
import time

# from modules.input_reader import VideoReader, ImageReader
from modules.draw import Plotter3d, draw_poses
from modules.parse_poses import parse_poses

import moviepy.editor as mpe
import librosa
import soundfile as sf

from argparse import ArgumentParser

def rotate_poses(poses_3d, R, t):
    R_inv = np.linalg.inv(R)
    for pose_id in range(len(poses_3d)):
        pose_3d = poses_3d[pose_id].reshape((-1, 4)).transpose()
        pose_3d[0:3, :] = np.dot(R_inv, pose_3d[0:3, :] - t)
        poses_3d[pose_id] = pose_3d.transpose().reshape(-1)

    return poses_3d

parser = ArgumentParser(description=''
                                    'Press esc to exit, "p" to (un)pause video or process next image.')

parser.add_argument('--joints', '-j', help='Mandatory path to joints numpy', type=str, required=True)
parser.add_argument('--music', '-m',
                    help='Path to music file',
                    type=str,default='')

parser.add_argument('--duration', '-d',
                    help='Duration of music',
                    type=str,default='')

args = parser.parse_args()

joints_path = args.joints
music_path = args.music
music_duration = args.duration

music_req = True
if music_path == '':
    music_req = False


canvas_3d = np.zeros((720, 1280, 3), dtype=np.uint8)
plotter = Plotter3d(canvas_3d.shape[:2])
canvas_3d_window_name = 'Canvas 3D'
cv2.namedWindow(canvas_3d_window_name)
cv2.setMouseCallback(canvas_3d_window_name, Plotter3d.mouse_callback)

file_path = None
if file_path is None:
    file_path = os.path.join('data', 'extrinsics.json')
with open(file_path, 'r') as f:
    extrinsics = json.load(f)
R = np.array(extrinsics['R'], dtype=np.float32)
t = np.array(extrinsics['t'], dtype=np.float32)

base_height = 256


delay = 1
esc_code = 27
p_code = 112
space_code = 32
mean_time = 0
is_video = False

with open(joints_path,'rb') as f:
    pose_data = np.load(f)

start = time.time()

counter = 0
for poses_3d in pose_data:

    print("working")

    stuff_val = np.array([0.5,0.5,0.5])

    part1 = poses_3d[:2]
    part2 = poses_3d[2:]

    pose_temp = np.vstack((part1,stuff_val))
    poses_3d = np.vstack((pose_temp,part2))

    poses_3d = poses_3d.reshape(1,19,3)

    # print(poses_3d.shape)
    # poses_3d.reshape

    edges = []

    

    if len(poses_3d):
        poses_3d = poses_3d.reshape(poses_3d.shape[0], 19, -1)[:, :, 0:3]
        edges = (Plotter3d.SKELETON_EDGES + 19 * np.arange(poses_3d.shape[0]).reshape((-1, 1, 1))).reshape((-1, 2))
    plotter.plot(canvas_3d, poses_3d, edges)
    # cv2.imshow(canvas_3d_window_name, canvas_3d)

    frame = canvas_3d

    cv2.imwrite('./temp_images/'+str(counter)+".jpg",canvas_3d)
    counter+=1

    # key = cv2.waitKey(100)

end_time = time.time()

print("Time taken: {}".format(end_time-start))

height,width = frame.shape[0], frame.shape[1]

video = cv2.VideoWriter('./rendered.mp4', 0x7634706d, 5, (width, height))

for i in range(50):
    file_name = './temp_images/'+str(i)+'.jpg'
    frame = cv2.imread(file_name)
    video.write(frame)
    os.remove(file_name)

if music_req == True:

    music_duration = int(music_duration)

    audio_clip = music_path
    y,sr = librosa.load(audio_clip,sr=16000)
    y = y[:16000*music_duration]
    sf.write('./temp_images/song.wav', y, 16000, 'PCM_24')

