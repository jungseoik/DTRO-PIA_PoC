from infer.people_count_api import process_video_ebc , process_video_ebc2 , process_image_ebc_dtro , process_image_ebc
# from infer.vqa_api import process_video
from infer.vqa_api_v2 import process_video
from infer.vqa_api_v3 import process_video_v3
import cv2
import os
from infer.vqa_smi_api import collect_subtitles_from_video ,create_video_with_subtitles
from env.config import PROMPT , PROMPT_V2
if __name__ == "__main__":
    template = """
    explain this image shortly
    """

    video_path = "/home/ws-internvl/DTRO/Crowd_People_Counting_Server_API/assets/sample44.mp4"
    folder_path = "/home/ws-internvl/DTRO/Crowd_People_Counting_Server_API/assets/sample_cnt"
 
    ## video ebc cnt - one video
    # process_video_ebc2(video_path = video_path, time_interval = 30)
    ## cnt only
    # process_image_ebc(folder_path)
    ## dot, dense
    # process_image_ebc_dtro(folder_path)
    ## only dense
    # process_image_ebc_dtro(folder_path, save_dot_map=False)
    ## subtitle video only one - process    
    # process_video(video_path, 30, template, lang="en")
    ## 
    process_video_v3(video_path, 30, PROMPT_V2)
