#!/usr/bin/env python3
"""Verify that the dedicated Paddle GPU environment can execute on GPU 0."""

from __future__ import annotations

import json

import paddle


def main() -> int:
    compiled_with_cuda = paddle.device.is_compiled_with_cuda()
    device_count = paddle.device.cuda.device_count() if compiled_with_cuda else 0
    if not compiled_with_cuda or device_count < 1:
        print(
            json.dumps(
                {
                    "ready": False,
                    "compiled_with_cuda": compiled_with_cuda,
                    "cuda_device_count": device_count,
                },
                indent=2,
            )
        )
        return 1
    paddle.device.set_device("gpu:0")
    tensor = paddle.to_tensor([1.0, 2.0, 3.0])
    value = float(tensor.sum())
    ready = value == 6.0
    print(
        json.dumps(
            {
                "ready": ready,
                "paddle_version": paddle.__version__,
                "compiled_cuda": paddle.version.cuda(),
                "compiled_cudnn": paddle.version.cudnn(),
                "cuda_device_count": device_count,
                "tensor_place": str(tensor.place),
            },
            indent=2,
        )
    )
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
