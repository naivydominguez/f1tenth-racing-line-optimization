"""Occupancy-grid map IO (ROS map_server / f1tenth_gym format: image + YAML)."""
import os
import numpy as np
import yaml
import cv2

from .geometry import signed_area


def read_map(yaml_path):
    """Returns dict(img (H,W) uint8 with row 0 = top of image, resolution, origin(x,y,yaw), ...)."""
    with open(yaml_path) as f:
        meta = yaml.safe_load(f)
    img_path = meta['image']
    if not os.path.isabs(img_path):
        img_path = os.path.join(os.path.dirname(os.path.abspath(yaml_path)), img_path)
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(img_path)
    if img.ndim == 3:
        img = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    meta['img'] = img
    meta['origin'] = [float(v) for v in meta['origin']]
    meta.setdefault('negate', 0)
    meta.setdefault('occupied_thresh', 0.65)
    meta.setdefault('free_thresh', 0.196)
    return meta


def free_mask(meta):
    """Boolean mask of free cells using the map's own thresholds (ROS map_server convention)."""
    img = meta['img'].astype(float)
    p = img / 255.0 if meta['negate'] else (255.0 - img) / 255.0  # occupancy probability
    # slam_toolbox writes unknown cells as 205 (p = 0.196), which is *below* its own
    # free_thresh of 0.25 -- never count those as free for planning.
    return p < min(meta['free_thresh'], 0.1)


def px_to_world(rows, cols, meta):
    H = meta['img'].shape[0]
    res = meta['resolution']
    ox, oy = meta['origin'][:2]
    x = ox + (np.asarray(cols) + 0.5) * res
    y = oy + (H - 1 - np.asarray(rows) + 0.5) * res
    return x, y


def world_to_px(x, y, meta):
    H = meta['img'].shape[0]
    res = meta['resolution']
    ox, oy = meta['origin'][:2]
    cols = (np.asarray(x) - ox) / res - 0.5
    rows = H - 1 - ((np.asarray(y) - oy) / res - 0.5)
    return rows, cols


def render_map(track, out_dir, name, res=0.05, pad=2.0, wall_px=3, style='gym'):
    """Rasterise a Track into <name>_map.png + <name>_map.yaml.

    style='gym'  : black outside the track (f1tenth_gym treats pixels <= 128 as walls)
    style='slam' : grey 'unknown' outside + black walls, like a slam_toolbox map
    """
    os.makedirs(out_dir, exist_ok=True)
    left, right = track.left_edge, track.right_edge
    allp = np.vstack([left, right])
    x0, y0 = allp.min(axis=0) - pad
    x1, y1 = allp.max(axis=0) + pad
    W = int(np.ceil((x1 - x0) / res))
    H = int(np.ceil((y1 - y0) / res))
    meta = dict(img=np.zeros((H, W), np.uint8), resolution=res, origin=[x0, y0, 0.0])

    def poly_px(p):
        r, c = world_to_px(p[:, 0], p[:, 1], meta)
        return np.round(np.column_stack([c, r]) * 16).astype(np.int32)  # 4 bits subpixel

    hw = 0.5 * wall_px * res
    left_o = track.xy + track.normal * (track.w_left + hw)[:, None]
    right_o = track.xy - track.normal * (track.w_right + hw)[:, None]
    outer, inner = (left_o, right_o) if abs(signed_area(left)) > abs(signed_area(right)) else (right_o, left_o)
    m_out = np.zeros((H, W), np.uint8)
    m_in = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m_out, [poly_px(outer)], 255, lineType=cv2.LINE_8, shift=4)
    cv2.fillPoly(m_in, [poly_px(inner)], 255, lineType=cv2.LINE_8, shift=4)
    corridor = (m_out > 0) & (m_in == 0)

    if style == 'slam':
        img = np.full((H, W), 205, np.uint8)
        img[corridor] = 254
    else:
        img = np.zeros((H, W), np.uint8)
        img[corridor] = 255
    # walls drawn just OUTSIDE the edges so the free corridor is exactly the CSV width
    half = 0.5 * wall_px * res
    for edge in (track.xy + track.normal * (track.w_left + half)[:, None],
                 track.xy - track.normal * (track.w_right + half)[:, None]):
        cv2.polylines(img, [poly_px(edge)], True, 0, thickness=wall_px, lineType=cv2.LINE_8, shift=4)

    png = f'{name}_map.png'
    cv2.imwrite(os.path.join(out_dir, png), img)
    meta_out = dict(image=png, resolution=float(res), origin=[float(x0), float(y0), 0.0],
                    negate=0, occupied_thresh=0.45, free_thresh=0.196)
    with open(os.path.join(out_dir, f'{name}_map.yaml'), 'w') as f:
        yaml.safe_dump(meta_out, f, default_flow_style=None, sort_keys=False)
    return os.path.join(out_dir, f'{name}_map.yaml')
