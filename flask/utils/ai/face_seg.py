from os import path as osp
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms

from utils.ai.src.model import BiSeNet


MODEL = osp.join(
    osp.dirname(__file__),
    "models",
    "79999_iter.pth"
)


def vis_parsing_maps(im, parsing_anno):

    # Colors for all 20 parts
    part_colors = [
        [255, 0, 0],
        [255, 85, 0],
        [255, 170, 0],
        [255, 0, 85],
        [255, 0, 170],
        [0, 255, 0],
        [85, 255, 0],
        [170, 255, 0],
        [0, 255, 85],
        [0, 255, 170],
        [0, 0, 255],
        [85, 0, 255],
        [170, 0, 255],
        [0, 85, 255],
        [0, 170, 255],
        [255, 255, 0],
        [255, 255, 85], 
        [255, 255, 170],
        [255, 0, 255],
        [255, 85, 255],
        [255, 170, 255],
        [0, 255, 255],
        [85, 255, 255],
        [170, 255, 255]
    ]

    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(
        src=vis_parsing_anno,
        dsize=None,
        fx=1,
        fy=1,
        interpolation=cv2.INTER_NEAREST
    )
    vis_parsing_anno_color = 255 + np.zeros(
        shape=(vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)
    )

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    vis_im = cv2.addWeighted(
        src1=cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR),
        alpha=0.4,
        src2=vis_parsing_anno_color,
        beta=0.6,
        gamma=0
    )

    return vis_im, vis_parsing_anno


def evaluate(im):

    # prep image
    if type(im) == str:
        im = cv2.cvtColor(cv2.imread(im), cv2.COLOR_BGR2RGB)
    im_resize = cv2.resize(im, (512, 512))

    # define model & transform
    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.load_state_dict(
        state_dict = torch.load(
            f=MODEL,
            map_location=torch.device('cpu'),
            weights_only=True
        )
    )
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    with torch.no_grad():
        
        t = to_tensor(im_resize)

        t = torch.unsqueeze(t, 0)
        out = net(t)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)

        return vis_parsing_maps(
            im_resize,
            parsing
        )


def save_inferenced(im, anno, save_path):

    cv2.imwrite(save_path, anno)
    cv2.imwrite(save_path, im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])


if __name__=='__main__':

    # define input image
    example_jpg = osp.join(
        osp.dirname(
            osp.dirname(
                osp.dirname(__file__)
            )
        ),
        "static",
        "portraits",
        "_example.jpg"        
    )

    # get extension of input image
    _, ext = osp.splitext(example_jpg)

    # get image and annotation
    im, anno = evaluate(example_jpg)

    # save result so we can visualize it
    save_inferenced(im, anno, example_jpg.replace(ext, f"_inferenced{ext}"))
