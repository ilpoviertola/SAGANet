import typing as tp

import torch

from pycocotools import mask as coco_mask_utils


ReturnFmts = tp.Literal["pt", "np"]


def rle_to_mask(
    rle_counts: tp.Union[str, bytes, tp.List[str], tp.List[bytes]],
    height: int,
    width: int,
    return_fmt: ReturnFmts = "pt",
):
    if not isinstance(rle_counts, list):
        rle_counts = [rle_counts]  # type: ignore

    detlist = []
    for rle in rle_counts:
        if isinstance(rle, str):
            rle = rle.encode()
        detection = {"size": [height, width], "counts": rle}
        detlist.append(detection)

    mask = coco_mask_utils.decode(detlist)
    binaryMask = mask.astype("bool")
    if return_fmt == "pt":
        return torch.tensor(binaryMask)
    elif return_fmt == "np":
        return binaryMask
    else:
        raise ValueError(f"Unknown return format {return_fmt}")


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    test_rle_str = [
        "gab14k73L3N2M8A=J3M2N3O0000O011O011N2N3N1N2N100O1N2N2L4Hd^h1",
        "P`a3",
    ]
    h = 256
    w = 454
    mask = rle_to_mask(test_rle_str, h, w)
    plt.imsave("mask.png", mask[..., -1].numpy())
    assert mask.shape == (h, w, 2)
