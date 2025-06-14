#!/usr/bin/env python3
import os
import sys
import argparse
import shutil
from PIL import Image

def process_image(src_path, dst_dir, inverse=False):
    """
    inverse=True  → 그레이스케일로 변환 후 픽셀 반전해서 저장
    inverse=False → 원본 그대로 복사
    """
    base = os.path.splitext(os.path.basename(src_path))[0]
    ext  = os.path.splitext(src_path)[1]
    if inverse:
        img = Image.open(src_path).convert('L')
        inv = Image.eval(img, lambda x: 255 - x)
        out_name = f"{base}_inverse{ext}"
        out_path = os.path.join(dst_dir, out_name)
        inv.save(out_path)
    else:
        out_name = f"{base}{ext}"
        out_path = os.path.join(dst_dir, out_name)
        shutil.copy(src_path, out_path)
    print(f"[+] {'Inverted' if inverse else 'Copied'} → {out_path}")

def main():
    parser = argparse.ArgumentParser(
        description="특정 epoch/batch/image 의 disp, pred, mask, rgb 프레임을 일괄 처리합니다"
    )
    parser.add_argument('-e', '--epoch', required=True, type=int, help="epoch 번호")
    parser.add_argument('-b', '--batch', required=True, type=int, help="batch 번호")
    parser.add_argument('-i', '--image', required=True, type=int, help="frame index (정수)")
    parser.add_argument('--no-inverse', dest='inverse', action='store_false',
                        help="pred 이미지를 반전하지 않고 원본만 복사")
    parser.set_defaults(inverse=True)
    args = parser.parse_args()

    src_dir = os.path.join("outputs", "frames",
                           f"epoch_{args.epoch}_batch_{args.batch}")
    if not os.path.isdir(src_dir):
        print(f"Error: source dir not found → {src_dir}")
        sys.exit(1)

    dst_dir = "output_sample"
    os.makedirs(dst_dir, exist_ok=True)

    idx = f"{args.image:02d}"
    prefixes = ["disp", "pred", "mask", "rgb"]
    for pre in prefixes:
        fname = f"{pre}_frame_{idx}.png"
        src_path = os.path.join(src_dir, fname)
        if not os.path.isfile(src_path):
            print(f"[!] not found, skip → {src_path}")
            continue
        # pred 만 inverse 옵션 적용, 나머지는 무조건 복사
        do_inverse = (pre == 'pred') and args.inverse
        process_image(src_path, dst_dir, inverse=do_inverse)

if __name__ == '__main__':
    main()
