import os.path
import shutil
# from . import faces_in_video_lib
import faces_in_video_lib

if __name__ == '__main__':
    vid_name = 'test3_short1'
    clip_path = f'ironvideo/example_video/{vid_name}.mp4'
    output_dir = f'ironvideo/example_video/{vid_name}'
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    vp = faces_in_video_lib.ProcessVideo(enable_codeformer=False)
    dashboard_data = vp.naive_process_video(clip_path, output_dir,
            enable_postprocess_split=True, enable_postprocess_merge=True)
    print(dashboard_data)