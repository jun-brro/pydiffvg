# apps/refine_svg.py
import os
import io
import tempfile
import torch
import numpy as np
import pydiffvg
import ttools.modules
import skimage
import skimage.io

gamma = 1.0

def refine_svg_in_memory(svg_content: str,
                         target_image: np.ndarray,
                         num_iter: int = 250,
                         use_lpips_loss: bool = False) -> str:
    """
    svg_content: SVG 파일 내용 (string)
    target_image: H×W×C numpy array, float in [0,1] or uint8
    num_iter: 최적화 반복 횟수
    use_lpips_loss: LPIPS 손실 사용 여부
    returns: 최종 refined SVG 문자열
    """
    # 1) 임시 SVG 파일로 쓰기
    tmp_svg = tempfile.NamedTemporaryFile('w', suffix='.svg', delete=False)
    tmp_svg.write(svg_content)
    tmp_svg.close()
    svg_path = tmp_svg.name

    # 2) target_image → torch tensor (1×3×H×W)
    if target_image.dtype != np.float32:
        target_image = target_image.astype(np.float32) / 255.0
    target = torch.from_numpy(target_image).to(torch.float32)
    # NHWC → NCHW, batch 차원 추가
    target = target.unsqueeze(0).permute(0, 3, 1, 2).to(pydiffvg.get_device()).pow(gamma)

    # 3) SVG → scene
    canvas_w, canvas_h, shapes, shape_groups = pydiffvg.svg_to_scene(svg_path)
    os.remove(svg_path)  # 더 이상 필요 없는 임시 파일

    # 4) LPIPS 셋업
    perception_loss = ttools.modules.LPIPS().to(pydiffvg.get_device())

    # 5) 최적화 변수 준비
    for p in shapes:
        p.points.requires_grad = True
    for g in shape_groups:
        g.fill_color.requires_grad = True

    points_optim = torch.optim.Adam([p.points for p in shapes], lr=1.0)
    color_optim  = torch.optim.Adam([g.fill_color for g in shape_groups], lr=0.01)

    render = pydiffvg.RenderFunction.apply

    # 6) 반복 최적화
    for t in range(num_iter):
        points_optim.zero_grad()
        color_optim.zero_grad()

        scene_args = pydiffvg.RenderFunction.serialize_scene(
            canvas_w, canvas_h, shapes, shape_groups)
        img = render(canvas_w, canvas_h, 2, 2, t, None, *scene_args)

        # alpha compositing
        img_rgb = img[:, :, 3:4] * img[:, :, :3] + \
                  torch.ones_like(img[:, :, :3]) * (1 - img[:, :, 3:4])

        # HWC → NCHW batch
        img_t = img_rgb.unsqueeze(0).permute(0, 3, 1, 2)

        # loss
        if use_lpips_loss:
            loss = perception_loss(img_t, target)
        else:
            loss = (img_t - target).pow(2).mean()

        loss.backward()
        points_optim.step()
        color_optim.step()

        # color clamp
        for g in shape_groups:
            g.fill_color.data.clamp_(0.0, 1.0)

    # 7) 최종 SVG를 또 임시 파일에 쓰고 읽어오기
    tmp_out = tempfile.NamedTemporaryFile('r', suffix='.svg', delete=False)
    tmp_out.close()
    pydiffvg.save_svg(tmp_out.name, canvas_w, canvas_h, shapes, shape_groups)

    with open(tmp_out.name, 'r') as f:
        final_svg = f.read()
    os.remove(tmp_out.name)

    return final_svg


# 예시 사용법
if __name__ == "__main__":
    # 파일 입출력 없이도 이렇게 바로 쓸 수 있다:
    svg_str = open("input.svg").read()
    img_np  = skimage.io.imread("input.png")  # 또는 numpy array
    refined_svg = refine_svg_in_memory(svg_str, img_np,
                                       num_iter=200,
                                       use_lpips_loss=True)
    print(refined_svg)  # 최종 SVG 코드를 stdout 으로 출력
