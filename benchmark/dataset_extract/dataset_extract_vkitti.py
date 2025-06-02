import os
import numpy as np
import os.path as osp
from PIL import Image
from tqdm import tqdm
import json
import glob
import shutil
import cv2
from natsort import natsorted
import argparse

from eval_utils import even_or_odd
from eval_utils import gen_json, get_sorted_files, copy_crop_files


def extract_vkitti(
    source_root,
    saved_dir="",
    sample_len=110,
    dataset_name="vkitti",
):
    """
    VKITTI 데이터셋을 평가용 포맷으로 추출
    
    Args:
        source_root: VKITTI 소스 데이터 경로 (vkitti_2.0.3_rgb, vkitti_2.0.3_depth가 있는 폴더)
        saved_dir: 저장할 디렉토리
        sample_len: 각 시퀀스에서 추출할 프레임 개수 (110 또는 500)
        dataset_name: 저장될 데이터셋 이름
    """
    rgb_root = osp.join(source_root, "vkitti_2.0.3_rgb")
    depth_root = osp.join(source_root, "vkitti_2.0.3_depth")
    
    # 평가용으로 분리된 데이터는 Scene20 사용
    eval_scene = "Scene20"
    
    # 각 시퀀스(conditions)와 카메라를 분리하여 개별 시퀀스로 취급
    sequences = []
    
    # Scene20에서 모든 condition 찾기
    scene_path = osp.join(rgb_root, eval_scene)
    if osp.exists(scene_path):
        conditions = sorted([d for d in os.listdir(scene_path) if osp.isdir(osp.join(scene_path, d))])
        
        for condition in conditions:
            for camera in ["Camera_0", "Camera_1"]:
                sequence_name = f"{eval_scene}_{condition}_{camera}"
                rgb_path = osp.join(rgb_root, eval_scene, condition, "frames", "rgb", camera)
                depth_path = osp.join(depth_root, eval_scene, condition, "frames", "depth", camera)
                
                if osp.exists(rgb_path) and osp.exists(depth_path):
                    # RGB 파일 목록 정렬
                    rgb_files = sorted([f for f in os.listdir(rgb_path) if f.endswith('.png')])
                    
                    # Depth 파일 목록 정렬
                    depth_files = sorted([f for f in os.listdir(depth_path) if f.endswith('.png')])
                    
                    # 최소 파일 개수로 제한
                    min_files = min(len(rgb_files), len(depth_files))
                    rgb_files = rgb_files[:min_files]
                    depth_files = depth_files[:min_files]
                    
                    if min_files > 0:
                        sequences.append({
                            "name": sequence_name,
                            "rgb_path": rgb_path,
                            "depth_path": depth_path,
                            "rgb_files": rgb_files,
                            "depth_files": depth_files,
                            "n_frames": min_files
                        })
                        print(f"시퀀스 '{sequence_name}'에서 {min_files}개 프레임 발견")
    
    if not sequences:
        print(f"평가용 데이터(Scene20)를 찾을 수 없습니다: {scene_path}")
        return
    
    # 각 시퀀스에서 프레임 추출
    for seq in tqdm(sequences, desc="시퀀스 처리 중"):
        seq_name = seq["name"]
        n_frames = seq["n_frames"]
        
        # 추출할 프레임 수가 실제 프레임 수보다 많으면 모두 사용
        if sample_len > n_frames or sample_len <= 0:
            frames_to_extract = n_frames
            step = 1
        else:
            frames_to_extract = sample_len
            # 균등하게 프레임 추출하기 위한 스텝 계산
            step = max(1, n_frames // frames_to_extract)
        
        print(f"시퀀스 '{seq_name}': 총 {n_frames} 프레임 중 {frames_to_extract}개 추출 (step={step})")
        
        # 디렉토리 생성
        os.makedirs(osp.join(saved_dir, dataset_name, seq_name, "rgb"), exist_ok=True)
        os.makedirs(osp.join(saved_dir, dataset_name, seq_name, "depth"), exist_ok=True)
        
        # 프레임 추출
        frames_extracted = 0
        for i in range(0, n_frames, step):
            if frames_extracted >= frames_to_extract:
                break
                
            rgb_file = seq["rgb_files"][i]
            depth_file = seq["depth_files"][i]
            
            # 입력 및 출력 경로
            img_path = osp.join(seq["rgb_path"], rgb_file)
            depth_path = osp.join(seq["depth_path"], depth_file)
            
            out_img_path = osp.join(saved_dir, dataset_name, seq_name, "rgb", rgb_file)
            out_depth_path = osp.join(saved_dir, dataset_name, seq_name, "depth", depth_file)
            
            # VKITTI 특화 처리를 위한 함수
            extract_vkitti_frame(
                img_path=img_path,
                depth_path=depth_path,
                out_img_path=out_img_path,
                out_depth_path=out_depth_path
            )
            
            frames_extracted += 1
        
        print(f"시퀀스 '{seq_name}': {frames_extracted}개 프레임 추출 완료")
    
    # 110 프레임 세트 JSON 생성
    out_json_path = osp.join(saved_dir, dataset_name, f"{dataset_name}_video.json")
    gen_json(
        root_path=osp.join(saved_dir, dataset_name),
        dataset=dataset_name,
        start_id=0,
        end_id=110,
        step=1,
        save_path=out_json_path
    )
    print(f"110 프레임 JSON 생성 완료: {out_json_path}")
    
    # 500 프레임 세트 JSON 생성 (프레임이 충분하다면)
    out_json_path = osp.join(saved_dir, dataset_name, f"{dataset_name}_video_500.json")
    gen_json(
        root_path=osp.join(saved_dir, dataset_name),
        dataset=dataset_name,
        start_id=0,
        end_id=500,
        step=1,
        save_path=out_json_path
    )
    print(f"500 프레임 JSON 생성 완료: {out_json_path}")


def extract_vkitti_frame(img_path, depth_path, out_img_path, out_depth_path):
    """
    VKITTI 프레임을 추출하고 평가 형식으로 변환하는 함수
    
    Args:
        img_path: 원본 RGB 이미지 경로
        depth_path: 원본 깊이 이미지 경로
        out_img_path: 출력할 RGB 이미지 경로
        out_depth_path: 출력할 깊이 이미지 경로
    """
    try:
        # RGB 이미지 읽기
        img = np.array(Image.open(img_path))
        
        # 이미지 크기 조정 (KITTI와 비슷하게 맞춤)
        height, width = img.shape[:2]
        height = even_or_odd(height)  # 짝수로 만들기
        width = even_or_odd(width)    # 짝수로 만들기
        img = img[:height, :width]    # 크기 조정
        
        # RGB 이미지 저장
        os.makedirs(osp.dirname(out_img_path), exist_ok=True)
        cv2.imwrite(out_img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))  # OpenCV는 BGR 형식으로 저장
        
        # 깊이 이미지 처리
        # 깊이 이미지를 변환하지 않고 그대로 복사
        os.makedirs(osp.dirname(out_depth_path), exist_ok=True)
        shutil.copyfile(depth_path, out_depth_path)
        
    except Exception as e:
        print(f"프레임 추출 중 오류 발생: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VKITTI 데이터셋을 평가용 포맷으로 추출")
    parser.add_argument("--source_root", type=str, default="./datasets/KITTI", 
                        help="VKITTI 소스 데이터 경로 (vkitti_2.0.3_rgb, vkitti_2.0.3_depth가 있는 폴더)")
    parser.add_argument("--saved_dir", type=str, default="./benchmark/datasets", 
                        help="데이터셋을 저장할 디렉토리")
    parser.add_argument("--sample_len", type=int, default=110, 
                        help="각 시퀀스에서 추출할 프레임 개수 (110 또는 500)")
    parser.add_argument("--dataset_name", type=str, default="vkitti", 
                        help="저장될 데이터셋 이름")
    
    args = parser.parse_args()
    
    extract_vkitti(
        source_root=args.source_root,
        saved_dir=args.saved_dir,
        sample_len=args.sample_len,
        dataset_name=args.dataset_name
    )