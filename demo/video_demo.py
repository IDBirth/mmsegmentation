# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

import cv2
import numpy as np
from mmengine.model.utils import revert_sync_batchnorm

from mmseg.apis import inference_model, init_model


def build_color_mask(frame: np.ndarray, result,
                     palette_bgr: np.ndarray) -> np.ndarray:
    """Build color segmentation mask in BGR format."""
    pred = result.pred_sem_seg.data.squeeze(0).detach().cpu().numpy()

    if pred.shape[:2] != frame.shape[:2]:
        pred = cv2.resize(
            pred.astype(np.uint8),
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_NEAREST)

    pred = pred.astype(np.int32)
    valid = (pred >= 0) & (pred < len(palette_bgr))
    pred_safe = np.where(valid, pred, 0)

    return palette_bgr[pred_safe].astype(np.uint8)


def main():
    parser = ArgumentParser()
    parser.add_argument('video', help='Video file or webcam id')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='cityscapes',
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--show', action='store_true', help='Whether to show draw result')
    parser.add_argument(
        '--show-wait-time', default=1, type=int, help='Wait time after imshow')
    parser.add_argument(
        '--output-file', default=None, type=str, help='Output video file path')
    parser.add_argument(
        '--output-fourcc',
        default='MJPG',
        type=str,
        help='Fourcc of the output video')
    parser.add_argument(
        '--output-fps', default=-1, type=int, help='FPS of the output video')
    parser.add_argument(
        '--output-height',
        default=-1,
        type=int,
        help='Frame height of the output video')
    parser.add_argument(
        '--output-width',
        default=-1,
        type=int,
        help='Frame width of the output video')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument(
        '--infer-every',
        type=int,
        default=25,
        help='Run model inference once every N frames. Default: 25')
    args = parser.parse_args()
    assert args.infer_every > 0, '--infer-every must be > 0'

    assert args.show or args.output_file, \
        'At least one output should be enabled.'

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)
    palette_bgr = np.array(model.dataset_meta['palette'], dtype=np.uint8)[:, ::-1]

    # build input video
    if args.video.isdigit():
        args.video = int(args.video)
    cap = cv2.VideoCapture(args.video)
    assert (cap.isOpened())
    input_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    input_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    input_fps = cap.get(cv2.CAP_PROP_FPS)

    # init output video
    writer = None
    output_height = None
    output_width = None
    if args.output_file is not None:
        fourcc = cv2.VideoWriter_fourcc(*args.output_fourcc)
        output_fps = args.output_fps if args.output_fps > 0 else input_fps
        output_height = args.output_height if args.output_height > 0 else int(
            input_height)
        output_width = args.output_width if args.output_width > 0 else int(
            input_width)
        writer = cv2.VideoWriter(args.output_file, fourcc, output_fps,
                                 (output_width, output_height), True)

    # start looping
    frame_id = 0
    cached_color_mask = None
    try:
        while True:
            flag, frame = cap.read()
            if not flag:
                break

            # Run inference every N frames and reuse last mask in-between.
            should_infer = (frame_id % args.infer_every == 0
                            or cached_color_mask is None
                            or cached_color_mask.shape[:2] != frame.shape[:2])
            if should_infer:
                result = inference_model(model, frame)
                cached_color_mask = build_color_mask(frame, result, palette_bgr)

            draw_img = cv2.addWeighted(frame, 1 - args.opacity,
                                       cached_color_mask, args.opacity, 0)
            frame_id += 1

            if args.show:
                cv2.imshow('video_demo', draw_img)
                key = cv2.waitKey(args.show_wait_time) & 0xFF
                if key == ord('q'):
                    break
            if writer:
                if draw_img.shape[0] != output_height or draw_img.shape[
                        1] != output_width:
                    draw_img = cv2.resize(draw_img,
                                          (output_width, output_height))
                writer.write(draw_img)
    finally:
        if writer:
            writer.release()
        cap.release()


if __name__ == '__main__':
    main()
