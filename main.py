from infer.people_count_api import process_video_ebc , process_video_ebc2 , process_image_ebc_dtro , process_image_ebc
# from infer.vqa_api import process_video
from infer.vqa_api_v2 import process_video
from infer.vqa_api_v3 import process_video_v3
from infer.vqa_api_v4 import  batch_process_videos
from infer.vqa_api_v5 import process_video2 , batch_process_videos2


import cv2
import os
from infer.vqa_smi_api import collect_subtitles_from_video ,create_video_with_subtitles
from env.config import PROMPT , PROMPT_V2 , PROMPT_V3
if __name__ == "__main__":
    template = """
    explain this image shortly
    """

    video_path = "/home/ws-internvl/DTRO/Crowd_People_Counting_Server_API/assets/sample44.mp4"
    folder_path = "/home/ws-internvl/DTRO/Crowd_People_Counting_Server_API/assets/sample_cnt"
    
    # folder_path = "/media/ws-internvl/5126-A2AE/영상분석(250708)/객실/317/상"
    # folder_path = "/media/ws-internvl/5126-A2AE/영상분석(250708)/객실/317/중"
    folder_path = "/media/ws-internvl/5126-A2AE/영상분석(250708)/객실/317/하"

    ## video ebc cnt - one video
    # process_video_ebc2(video_path = video_path, time_interval = 30)
    ## image folder cnt only
    # process_image_ebc(folder_path)
    ## dot, dense
    # process_image_ebc_dtro(folder_path)
    ## only dense
    # process_image_ebc_dtro(folder_path, save_dot_map=False)
    
    ## only subtitle video only one - process    
    # process_video(video_path, 30, template, lang="en")
    ## subtitle & alarm video only one - 
    ##
    folder_path = "/home/ws-internvl/DTRO/쓰러짐"

    # folder_path = "/home/ws-internvl/anaconda3/docs/사고영상_샘플추출_result/쓰러짐"
    # folder_path = "/home/ws-internvl/anaconda3/docs/사고영상_샘플추출_result/정상상황"
    batch_process_videos(folder_path, 30, PROMPT_V3, output_root_dir="/home/ws-internvl/DTRO/Crowd_People_Counting_Server_API/result/last_result/prompt_v3")
    batch_process_videos(folder_path, 30, PROMPT_V2 , output_root_dir="/home/ws-internvl/DTRO/Crowd_People_Counting_Server_API/result/last_result/prompt_v2")
    # process_video_v3(video_path, 30, PROMPT_V2)
    # process_video_v3(video_path, 30, PROMPT_V3)


    # folder_path = "/home/ws-internvl/anaconda3/docs/사고영상_샘플추출_kor/정상상황"
    # batch_process_videos2(folder_path ,30, question = template, lang= "kor" ,result_root_dir="/home/ws-internvl/DTRO/Crowd_People_Counting_Server_API/result/kor_explain_정상")
    # folder_path = "/home/ws-internvl/anaconda3/docs/사고영상_샘플추출_kor/쓰러짐"
    # batch_process_videos2(folder_path ,30, question = template, lang= "kor" ,result_root_dir="/home/ws-internvl/DTRO/Crowd_People_Counting_Server_API/result/kor_explain_쓰러짐")

