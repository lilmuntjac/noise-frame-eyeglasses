from pathlib import Path
from PIL import Image

def main(args):
    root_dir = Path(args.in_dir)
    image_list = list((root_dir).glob('*.png'))
    images = [Image.open(x) for x in image_list]
    # first 16 images form a 4x4 grid
    width, _ = images[0].size
    total_width = width*4
    image_out = Image.new('RGB', (total_width, total_width))
    for i in range(4):
        for j in range(4):
            image_out.paste(images[i*4+j], (i*width, j*width))

    image_out.save(args.output)
    
def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-i", "--in-dir", type=str, default="./img", help="input folder for image")
    parser.add_argument("-o", "--output", type=str, default="./img.png", help="output image path")


    return parser

if __name__ == '__main__':
    args = get_args().parse_args()
    main(args)