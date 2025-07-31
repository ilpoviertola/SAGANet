"""
This script downloads and extracts 5 second video clips from the Segmented Music Solos (SMS) dataset.
First download the raw data following the instructions in the following datasets:
MUSIC21 (https://github.com/roudimit/MUSIC_dataset)
AVSSBench (https://github.com/OpenNLPLab/AVSBench)
Solos (https://github.com/JuanFMontesinos/Solos)
URMP (https://labsites.rochester.edu/air/projects/URMP.html)
"""

import argparse
from pathlib import Path
import requests
import typing as tp
from shutil import move, copy2

from tqdm import tqdm
import ffmpeg
import pandas as pd


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
        "--urmp",
        type=str,
        required=True,
        help="Path to the URMP dataset directory (not used in this script, but required for consistency)",
    )
    parser.add_argument(
        "--move_files",
        action="store_true",
        help="Move files instead of copying them",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="Output directory for datasets",
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
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist.")
    return pd.read_csv(file_path)


def read_tsv(file_path: Path):
    if not file_path.exists():
        raise FileNotFoundError(f"File {file_path} does not exist.")
    return pd.read_csv(file_path, sep="\t")


def download_files(output_dir: Path):
    file_list = download(output_dir / "files.txt")
    files_to_download = file_list.read_text().splitlines()
    files_to_download = [f for f in files_to_download if not f.endswith(".pth")]
    print(f"Downloading {len(files_to_download)} files...")
    for file_name in tqdm(
        files_to_download,
        desc="Downloading files",
        unit="file",
        total=len(files_to_download),
    ):
        file_path = output_dir / file_name
        if not file_path.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)
            download(file_path, f"{SAGANET_PUB_URL}/{file_name}")


def handle_sms(
    solos_path: Path,
    music21_path: Path,
    avssbench_path: Path,
    output_dir: Path,
    act: tp.Callable,
):
    assert (
        output_dir / "sms" / "metadata.csv"
    ).exists(), "Metadata file not found in the SMS directory."

    metadata = read_csv(output_dir / "sms" / "metadata.csv")
    metadata = metadata.reset_index()

    output_dir = output_dir / "sms"
    for i, row in metadata.iterrows():
        if i == 5:
            break
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
        fp = fp[0] if fp else None  # type: ignore
        if not fp:
            fn = row["filename"]
            fp = list(inp.rglob(f"{fn}.mp4"))
            fp = fp[0] if fp else None  # type: ignore
            processed = True
        if not fp:
            print(f"File not found for {row['YTID']} ({row['filename']})")
            continue
        if not processed:
            handle_video_file(
                fp,  # type: ignore
                output_dir / "data" / row["redefined_label"],
                row["start_msec"],
                row["end_msec"],
            )
        else:
            act(fp, output_dir / "data" / row["redefined_label"] / fp.name)  # type: ignore


def handle_urmp(
    urmp_path: Path,
    output_dir: Path,
    act: tp.Callable,
):
    assert (
        output_dir / "urmp" / "metadata.tsv"
    ).exists(), "Metadata file not found in the URMP directory."

    metadata = read_tsv(output_dir / "urmp" / "metadata.tsv")
    metadata = metadata.reset_index()

    output_dir = output_dir / "urmp"
    for i, row in metadata.iterrows():
        if i == 5:
            break
        processed = False
        p_dir, v_id = row["video_id"].split("/")
        matches = list(urmp_path.rglob(f"{v_id}.mp4"))
        if matches:
            assert len(matches) == 1, f"Multiple matches found for {v_id}"
            fp = matches[0]
            processed = True
        else:
            v_id = "_".join(v_id.split("_")[:-2])
            matches = list(urmp_path.rglob(f"{v_id}.mp4"))
            if matches:
                assert len(matches) == 1, f"Multiple matches found for {v_id}"
                fp = matches[0]
            else:
                print(f"File not found for {row['video_id']}")
                continue
        if not processed:
            handle_video_file(
                fp,
                output_dir / "data" / p_dir,
                row["s_ts"],
                row["e_ts"],
            )
        else:
            act(fp, output_dir / "data" / p_dir / fp.name)


def main():
    args = get_args()
    solos_path = Path(args.solos)
    assert solos_path.exists(), f"Solos path {solos_path} does not exist."
    music21_path = Path(args.music21)
    assert music21_path.exists(), f"Music21 path {music21_path} does not exist."
    avssbench_path = Path(args.avssbench)
    assert avssbench_path.exists(), f"AVSSBench path {avssbench_path} does not exist."
    urmp_path = Path(args.urmp)
    assert urmp_path.exists(), f"URMP path {urmp_path} does not exist."
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download SMS/URMP metadata and generated masks
    download_files(output_dir)

    # Create symlinks or move the files
    move_files = args.move_files
    if move_files:
        print("NOTE: We are moving files instead of copying them!!")
        act = move
    else:
        act = copy2

    handle_sms(
        solos_path,
        music21_path,
        avssbench_path,
        output_dir,
        act=act,
    )

    handle_urmp(
        urmp_path,
        output_dir,
        act=act,
    )


if __name__ == "__main__":
    main()
