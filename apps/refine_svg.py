# apps/refine_svg.py
import os
import io
import tempfile
import torch
import numpy as np
import pydiffvg
import skimage
import skimage.io

gamma = 1.0

def refine_svg_in_memory(svg_content: str,
                         target_image: np.ndarray,
                         num_iter: int = 250) -> str:
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

    # 3.5) target 해상도를 canvas 해상도에 맞추기
    # torch.nn.functional.interpolate를 사용해 (H,W)를 (canvas_h,canvas_w)로 보간
    import torch.nn.functional as F
    # target: [1, 3, H_target, W_target]
    if target.shape[2] != canvas_h or target.shape[3] != canvas_w:
        target = F.interpolate(
            target,
            size=(canvas_h, canvas_w),
            mode='bilinear',
            align_corners=False
        )
    # 5) 최적화 변수 준비
    points_vars = []
    for shape in shapes:
        # C++ 확장 타입인 Rect 같은 경우 getattr이 제대로 동작하지 않을 수 있으므로,
        # 직접 예외처리로 걸러낸다.
        try:
            pts = shape.points
        except AttributeError:
            # points 속성이 없으면 건너뜀
            continue
        if isinstance(pts, torch.Tensor):
            pts.requires_grad = True
            points_vars.append(pts)

    color_vars = []
    for group in shape_groups:
        col = getattr(group, 'fill_color', None)
        if isinstance(col, torch.Tensor):
            col.requires_grad = True
            color_vars.append(col)
    
    points_optim = torch.optim.Adam(points_vars, lr=1.0)
    color_optim  = torch.optim.Adam(color_vars,  lr=0.01)

    render = pydiffvg.RenderFunction.apply

    # 6) 반복 최적화
    # 6) 반복 최적화
    for t in range(num_iter):
        # 1) gradient 초기화
        points_optim.zero_grad()
        color_optim.zero_grad()
    
        # 2) scene serialization & rendering
        scene_args = pydiffvg.RenderFunction.serialize_scene(
            canvas_w, canvas_h, shapes, shape_groups)
        img = render(canvas_w, canvas_h, 2, 2, t, None, *scene_args)
    
        # 3) alpha compositing
        img_rgb = img[:, :, 3:4] * img[:, :, :3] + \
                  torch.ones_like(img[:, :, :3]) * (1 - img[:, :, 3:4])
    
        # 4) HWC → NCHW batch 차원 추가
        img_t = img_rgb.unsqueeze(0).permute(0, 3, 1, 2)
    
        # 5) 손실 계산
        loss = (img_t - target).pow(2).mean()
    
        # 6) 역전파 & 업데이트
        loss.backward()
        points_optim.step()
        color_optim.step()
    
        # 7) color_vars에 담긴 텐서만 [0,1] 범위로 clamp
        for col in color_vars:
            col.data.clamp_(0.0, 1.0)

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
