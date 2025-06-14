#!/usr/bin/env python3
"""
test_rgb_save.py

KITTI vKITTI RGB 이미지를 다양한 모드로 저장해보는 스크립트
- noswap  : OpenCV(BGR) 데이터를 그대로 PIL에 RGB로 저장
- swap    : BGR -> RGB 변환 후 저장
- pilopen : PIL.Image.open 으로 직접 읽어 저장

Usage:
    python test_rgb_save.py
"""
import os
import cv2
from PIL import Image


def main():
    # 원본 이미지 경로
    img_path = "./datasets/KITTI/vkitti_2.0.3_rgb/Scene01/15-deg-left/frames/rgb/Camera_1/rgb_00422.jpg"
    # 결과 저장 디렉토리
    save_dir = "outputs/test_rgb_modes"
    os.makedirs(save_dir, exist_ok=True)

    # 1) OpenCV로 BGR 컬러로 읽기
    bgr = cv2.imread(img_path)
    if bgr is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    # --- a) noswap: BGR 데이터를 RGB로 잘못 해석해 저장 (색상 왜곡 확인용) ---
    noswap_pil = Image.fromarray(bgr)
    noswap_pil.save(os.path.join(save_dir, "rgb_00422_noswap.png"))

    # --- b) swap: BGR -> RGB 변환 후 저장 (올바른 색상) ---
    rgb = bgr[..., ::-1]  # BGR -> RGB
    swap_pil = Image.fromarray(rgb)
    swap_pil.save(os.path.join(save_dir, "rgb_00422_swap.png"))

    # --- c) PIL.Image.open 로 직접 읽어 저장 (기본 RGB) ---
    pil_loaded = Image.open(img_path).convert("RGB")
    pil_loaded.save(os.path.join(save_dir, "rgb_00422_pilopen.png"))

    print(f"Saved test images to '{save_dir}'")


if __name__ == "__main__":
    main()
