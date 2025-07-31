"""
This script downloads and extracts 5 second video clips from the Segmented Music Solos (SMS) dataset.
First download the raw data following the instructions in the following datasets:
MUSIC21 (https://github.com/roudimit/MUSIC_dataset)
AVSSBench (https://github.com/OpenNLPLab/AVSBench)
Solos (https://github.com/JuanFMontesinos/Solos)
"""

import argparse
from pathlib import Path
import requests
from shutil import move, copy2

from tqdm import tqdm
import ffmpeg


SAGANET_PUB_URL = (
    "https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/saganet_public"
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--solos",
        type=str,
        required=True,
        help="Path to the Solos dataset directory",
    )
    parser.add_argument(
        "--music21",
        type=str,
        required=True,
        help="Path to the MUSIC21 dataset directory",
    )
    parser.add_argument(
        "--avssbench",
        type=str,
        required=True,
        help="Path to the AVSSBench dataset directory",
    )
    parser.add_argument(
        "--move_files",
        action="store_true",
        help="Move files instead of copying them",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./sms",
        help="Output directory for SMS clips",
    )
    return parser.parse_args()


def download(
    output_path: Path, link: str = SAGANET_PUB_URL, show_progress: bool = False
):
    if output_path.exists():
        print(f"File {output_path} already exists, skipping download.")
        return output_path
    r = requests.get(link, stream=True)
    total_size = int(r.headers.get("content-length", 0))
    block_size = 1024
    t = tqdm(total=total_size, unit="iB", unit_scale=True, disable=not show_progress)
    with open(output_path, "wb") as f:
        for data in r.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()
    return output_path


def handle_video_file(video_path: Path, out_path: Path, start_msec: int, end_msec: int):
    duration_sec = (end_msec - start_msec) / 1000
    out_clip = out_path / f"{video_path.stem}_{start_msec}_{end_msec}.mp4"
    stream = ffmpeg.input(video_path.as_posix(), ss=start_msec / 1000)
    stream = ffmpeg.output(
        stream.video,
        stream.audio,
        out_clip.as_posix(),
        t=duration_sec,
        loglevel="quiet",
    )
    stream.run(overwrite_output=True)
    return out_clip


def read_csv(file_path: Path):
    import pandas as pd

    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist.")
    return pd.read_csv(file_path)


def main():
    args = get_args()
    solos_path = Path(args.solos)
    assert solos_path.exists(), f"Solos path {solos_path} does not exist."
    music21_path = Path(args.music21)
    assert music21_path.exists(), f"Music21 path {music21_path} does not exist."
    avssbench_path = Path(args.avssbench)
    assert avssbench_path.exists(), f"AVSSBench path {avssbench_path} does not exist."
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download SMS metadata and generated masks
    file_list = download(output_dir / "files.txt")
    files_to_download = file_list.read_text().splitlines()
    print(f"Downloading {len(files_to_download)} files...")
    for file_name in tqdm(
        files_to_download,
        desc="Downloading SMS",
        unit="file",
        total=len(files_to_download),
    ):
        file_path = output_dir / file_name
        if not file_path.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)
            download(file_path, f"{SAGANET_PUB_URL}/{file_name}")
    assert (
        output_dir / "sms" / "metadata.csv"
    ).exists(), "Metadata file not found in the SMS directory."

    # Create symlinks or move the files
    move_files = args.move_files
    if move_files:
        print("NOTE: We are moving files instead of copying them!!")
        act = move
    else:
        act = copy2
    metadata = read_csv(output_dir / "sms" / "metadata.csv")
    metadata = metadata.reset_index()

    output_dir = output_dir / "sms"
    for _, row in metadata.iterrows():
        if row["subset"] == "solos":
            inp = solos_path
        elif row["subset"] == "avs-semantic":
            inp = avssbench_path
        elif row["subset"] == "music21":
            inp = music21_path
        else:
            print(f"Unknown subset {row['subset']} in metadata.")
            continue
        processed = False
        fn = row["YTID"]
        fp = list(inp.rglob(f"{fn}.mp4"))
        fp = fp[0] if fp else None
        if not fp:
            fn = row["filename"]
            fp = list(inp.rglob(f"{fn}.mp4"))
            fp = fp[0] if fp else None
            processed = True
        if not fp:
            print(f"File not found for {row['YTID']} ({row['filename']})")
            continue
        if not processed:
            handle_video_file(
                fp,
                output_dir / "data" / row["redefined_label"],
                row["start_msec"],
                row["end_msec"],
            )
        else:
            act(fp, output_dir / fp.name)


if __name__ == "__main__":
    main()
