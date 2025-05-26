"""
Scream: python painterly_rendering.py imgs/scream.jpg --num_paths 2048 --max_width 4.0
Fallingwater: python painterly_rendering.py imgs/fallingwater.jpg --num_paths 2048 --max_width 4.0
Fallingwater: python painterly_rendering.py imgs/fallingwater.jpg --num_paths 2048 --max_width 4.0 --use_lpips_loss
Baboon: python painterly_rendering.py imgs/baboon.png --num_paths 1024 --max_width 4.0 --num_iter 250
Baboon Lpips: python painterly_rendering.py imgs/baboon.png --num_paths 1024 --max_width 4.0 --num_iter 500 --use_lpips_loss
Kitty: python painterly_rendering.py imgs/kitty.jpg --num_paths 1024 --use_blob
"""
import pydiffvg
import torch
import skimage
import skimage.io
import random
import ttools.modules
import argparse
import math
import subprocess

pydiffvg.set_print_timing(True)

def noop(*args, **kwargs):
    pass

gamma = 1.0

def main(args):
    # Use GPU if available
    pydiffvg.set_use_gpu(torch.cuda.is_available())
    
    perception_loss = ttools.modules.LPIPS().to(pydiffvg.get_device())
    
    target = torch.from_numpy(skimage.io.imread(args.target)).to(torch.float32) / 255.0
    target = target.pow(gamma)
    target = target.to(pydiffvg.get_device())
    target = target.unsqueeze(0)
    target = target.permute(0, 3, 1, 2) # NHWC -> NCHW

    canvas_width, canvas_height = target.shape[3], target.shape[2]
    num_paths = args.num_paths
    max_width = args.max_width
    
    random.seed(1234)
    torch.manual_seed(1234)
    
    shapes = []
    shape_groups = []
    if args.use_blob:
        for i in range(num_paths):
            num_segments = random.randint(3, 5)
            num_control_points = torch.zeros(num_segments, dtype=torch.int32) + 2
            points = []
            p0 = (random.random(), random.random())
            points.append(p0)
            for j in range(num_segments):
                radius = 0.05
                p1 = (p0[0] + radius * (random.random() - 0.5),
                      p0[1] + radius * (random.random() - 0.5))
                p2 = (p1[0] + radius * (random.random() - 0.5),
                      p1[1] + radius * (random.random() - 0.5))
                points.append(p1)
                points.append(p2)
                if j < num_segments - 1:
                    p3 = (p2[0] + radius * (random.random() - 0.5),
                          p2[1] + radius * (random.random() - 0.5))
                    points.append(p3)
                    p0 = p3
            points = torch.tensor(points)
            points[:, 0] *= canvas_width
            points[:, 1] *= canvas_height
            path = pydiffvg.Path(num_control_points=num_control_points,
                                 points=points,
                                 stroke_width=torch.tensor(1.0),
                                 is_closed=True)
            shapes.append(path)
            path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(shapes) - 1]),
                                             fill_color=torch.tensor([random.random(),
                                                                      random.random(),
                                                                      random.random(),
                                                                      random.random()]))
            shape_groups.append(path_group)
    else:
        for i in range(num_paths):
            num_segments = random.randint(1, 3)
            num_control_points = torch.zeros(num_segments, dtype=torch.int32) + 2
            points = []
            p0 = (random.random(), random.random())
            points.append(p0)
            for j in range(num_segments):
                radius = 0.05
                p1 = (p0[0] + radius * (random.random() - 0.5),
                      p0[1] + radius * (random.random() - 0.5))
                p2 = (p1[0] + radius * (random.random() - 0.5),
                      p1[1] + radius * (random.random() - 0.5))
                p3 = (p2[0] + radius * (random.random() - 0.5),
                      p2[1] + radius * (random.random() - 0.5))
                points.append(p1)
                points.append(p2)
                points.append(p3)
                p0 = p3
            points = torch.tensor(points)
            points[:, 0] *= canvas_width
            points[:, 1] *= canvas_height
            path = pydiffvg.Path(num_control_points=num_control_points,
                                 points=points,
                                 stroke_width=torch.tensor(1.0),
                                 is_closed=False)
            shapes.append(path)
            path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(shapes) - 1]),
                                             fill_color=None,
                                             stroke_color=torch.tensor([random.random(),
                                                                         random.random(),
                                                                         random.random(),
                                                                         random.random()]))
            shape_groups.append(path_group)
    
    # Serialize scene for initial render and optimization
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_width, canvas_height, shapes, shape_groups)
    
    render = pydiffvg.RenderFunction.apply
    img = render(canvas_width, canvas_height, 2, 2, 0, None, *scene_args)
    pydiffvg.imwrite(img.cpu(), 'results/painterly_rendering/init.png', gamma=gamma)

    # Prepare variables for optimization
    points_vars = []
    stroke_width_vars = []
    color_vars = []
    for path in shapes:
        path.points.requires_grad = True
        points_vars.append(path.points)
    if not args.use_blob:
        for path in shapes:
            path.stroke_width.requires_grad = True
            stroke_width_vars.append(path.stroke_width)
    else:
        for group in shape_groups:
            group.fill_color.requires_grad = True
            color_vars.append(group.fill_color)
    if not args.use_blob:
        for group in shape_groups:
            group.stroke_color.requires_grad = True
            color_vars.append(group.stroke_color)
    
    # Optionally disable PNG output
    if args.only_svg:
        pydiffvg.imwrite = noop

    # Optimizers
    points_optim = torch.optim.Adam(points_vars, lr=1.0)
    width_optim = torch.optim.Adam(stroke_width_vars, lr=0.1) if stroke_width_vars else None
    color_optim = torch.optim.Adam(color_vars, lr=0.01)

    # Adam iterations
    for t in range(args.num_iter):
        print('iteration:', t)
        points_optim.zero_grad()
        if width_optim: width_optim.zero_grad()
        color_optim.zero_grad()

        scene_args = pydiffvg.RenderFunction.serialize_scene(
            canvas_width, canvas_height, shapes, shape_groups)
        img = render(canvas_width, canvas_height, 2, 2, t, None, *scene_args)
        img = img[:, :, 3:4] * img[:, :, :3] + \
              torch.ones_like(img[:, :, :3]) * (1 - img[:, :, 3:4])
        pydiffvg.imwrite(img.cpu(), f'results/painterly_rendering/iter_{t}.png', gamma=gamma)
        img = img.unsqueeze(0).permute(0, 3, 1, 2)

        if args.use_lpips_loss:
            loss = perception_loss(img, target) + (img.mean() - target.mean()).pow(2)
        else:
            loss = (img - target).pow(2).mean()
        print('render loss:', loss.item())

        loss.backward()
        points_optim.step()
        if width_optim: width_optim.step()
        color_optim.step()

        # Clamp values
        for path in shapes:
            path.stroke_width.data.clamp_(1.0, max_width)
        for group in shape_groups:
            if args.use_blob:
                group.fill_color.data.clamp_(0.0, 1.0)
            else:
                group.stroke_color.data.clamp_(0.0, 1.0)

        # Save periodic SVG snapshots
        if t % 10 == 0 or t == args.num_iter - 1:
            pydiffvg.save_svg(f'results/painterly_rendering/iter_{t}.svg',
                              canvas_width, canvas_height, shapes, shape_groups)
    
    if args.output:
        pydiffvg.save_svg(args.output,
                          canvas_width, canvas_height, shapes, shape_groups)

    if not args.only_svg:
        subprocess.call([
            "ffmpeg", "-framerate", "24", "-i",
            "results/painterly_rendering/iter_%d.png", "-vb", "20M",
            "results/painterly_rendering/out.mp4"
        ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("target", help="target image path")
    parser.add_argument("--num_paths", type=int, default=512)
    parser.add_argument("--max_width", type=float, default=2.0)
    parser.add_argument("--use_lpips_loss", action='store_true')
    parser.add_argument("--num_iter", type=int, default=500)
    parser.add_argument("--use_blob", action='store_true')
    parser.add_argument("--only_svg", action="store_true",
                        help="Skip PNG/video and only produce SVG")
    parser.add_argument("--output", type=str, default=None,
                        help="If set, write final SVG to this path")
    args = parser.parse_args()
    main(args)
