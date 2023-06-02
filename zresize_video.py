import os
import cv2
from tqdm import tqdm
from multiprocess.pool import Pool
from typing import Callable, Iterable, Optional

H = 480
W = 856

class TaskRunner:
    def __init__(self, pool_size=32) -> None:
        self.pool = Pool(pool_size)

    @staticmethod
    def __wrapper(func: Callable):
        def _wfunc(zipped_kwargs):
            return func(**zipped_kwargs)
        return _wfunc

    def run(self, func: Callable, args_iter: Iterable, *, total:Optional[int]=None):
        results = list(tqdm(self.pool.imap(self.__wrapper(func), args_iter), total=total))
        self.pool.terminate()
        return results

src_dirs = [
    '/medical-data/zsxm/运动热量估计/eev3/一口气法',
    '/medical-data/zsxm/运动热量估计/eev3/混合室法',
]
dst_dirs = [
    '/medical-data/zsxm/运动热量估计/eev3/resized/一口气法',
    '/medical-data/zsxm/运动热量估计/eev3/resized/混合室法',
]

def split_video(src_dir, dst_dir, video_name):
    if os.path.splitext(video_name)[1].lower() != '.mp4':
        return
    os.makedirs(dst_dir, exist_ok=True)
    # Open the video in read mode
    video_capture = cv2.VideoCapture(os.path.join(src_dir, video_name))
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    # 创建VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(os.path.join(dst_dir, video_name), fourcc, fps, (W, H))
    # Iterate over each frame of the video
    while True:
        success, frame = video_capture.read()
        # If there are no more frames, break out of the loop
        if not success:
            break
        frame = cv2.resize(frame, (W, H))
        out.write(frame)
    # Release the video capture object
    video_capture.release()
    out.release()

for src_dir, dst_dir in zip(src_dirs, dst_dirs):
    kwargs = [{'src_dir':src_dir, 'dst_dir':dst_dir, 'video_name':vn} for vn in sorted(os.listdir(src_dir))]
    res = TaskRunner().run(split_video, kwargs, total=len(kwargs))
