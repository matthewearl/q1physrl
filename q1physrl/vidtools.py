import pathlib
import sys

import cv2
import numpy as np
import PIL
import PIL.ImageDraw
import PIL.ImageFont
from matplotlib import cm
from tqdm import tqdm

from q1physrl import analyse


def _draw_speed_text(a, speed):
    image = PIL.Image.fromarray(a)
    draw = PIL.ImageDraw.Draw(image)
    font = PIL.ImageFont.truetype("/usr/share/fonts/truetype/msttcorefonts/Arial_Bold.ttf", 28)
    draw.text((10, 0), f"{int(speed)} ups", (255, 255, 255), font=font)

    return np.array(image)


def _draw_speed_bar(speed, shape, border=2, max_speed=700):
    viridis = cm.get_cmap('hot', shape[1])

    g = viridis(np.linspace(0, 1, shape[1])) * 255
    g[np.linspace(0, 700, shape[1]) > speed] = [0, 0, 0, 128]
    g = np.stack([g] * shape[0], axis=0)
    
    im = np.empty((shape[0] + border * 2, shape[1] + border * 2, 4), dtype=np.uint8)
    im[:, :] = np.array([0, 0, 0, 255])
    
    im[border:-border, border:-border] = g

    return im.astype(np.uint8)


def rgba_to_bgra(im):
    bgr = np.flip(im[:, :, :3], axis=2)
    alpha = im[:, :, 3]
    return np.concatenate([bgr, alpha[:, :, None]], axis=2)


def make_speed_anim():
    demo_file_path, output_dir = [pathlib.Path(x) for x in sys.argv[1:]]

    anim_fps = 60
    shape = (32, 256)

    times, origins, yaws = analyse.parse_demo(demo_file_path)

    vels = np.diff(origins, axis=0) / np.diff(times)[:, None]
    speeds = np.linalg.norm(vels[:, :2], axis=1)
    
    output_dir.mkdir(exist_ok=True, parents=True)
    frame_times = np.arange(times[0] * anim_fps, times[-1] * anim_fps) / anim_fps
    frame_speeds = np.interp(frame_times, times[:-1], speeds)
    for i, s in enumerate(tqdm(frame_speeds)):
        frame_im = _draw_speed_text(_draw_speed_bar(s, shape), s)
        
        if not cv2.imwrite(str(output_dir / f"{i:05d}.png"), rgba_to_bgra(frame_im)):
            raise Exception("couldn't save image")

