
from PIL import Image
import os
import sys

def invert_grayscale(input_path):
    # 1) 이미지 열기 → 그레이스케일 변환
    img = Image.open(input_path).convert('L')
    # 2) 픽셀 반전
    inverted = Image.eval(img, lambda x: 255 - x)
    # 3) 저장 경로 생성 (예: foo.jpg → foo_반전버전.jpg)
    base, ext = os.path.splitext(input_path)
    out_path = f"{base}_inverse{ext}"
    # 4) 저장
    inverted.save(out_path)
    print(f"Saved inverted image → {out_path}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python invert_gray.py <이미지_경로>")
        sys.exit(1)
    invert_grayscale(sys.argv[1])
