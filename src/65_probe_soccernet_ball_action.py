#!/usr/bin/env python3

import argparse
import inspect
import json
from pathlib import Path

import SoccerNet
from SoccerNet.Downloader import SoccerNetDownloader


BALL_SPLIT_FILES = {
    "train": "BallTrain.json",
    "valid": "BallValid.json",
    "test": "BallTest.json",
    "challenge": "BallChallenge.json",
}


def count_games(split_payload):
    count = 0
    for seasons in split_payload.values():
        for games in seasons.values():
            count += len(games)
    return count


def first_games(split_payload, limit=3):
    rows = []
    for league, seasons in split_payload.items():
        for season, games in seasons.items():
            for game in games:
                rows.append((league, season, game))
                if len(rows) >= limit:
                    return rows
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Inspect local SoccerNet Ball Action Spotting metadata without downloading data."
    )
    parser.add_argument(
        "--local-directory",
        default="data/open/soccernet",
        help="Local directory that would be used by SoccerNetDownloader.",
    )
    args = parser.parse_args()

    package_root = Path(SoccerNet.__file__).parent
    data_root = package_root / "data"

    print("SOCCERNET BALL ACTION SPOTTING PROBE")
    print("------------------------------------")
    print(f"SoccerNet package: {package_root}")
    print(f"SoccerNet version: {getattr(SoccerNet, '__version__', 'unknown')}")
    print("")

    for split, filename in BALL_SPLIT_FILES.items():
        path = data_root / filename
        payload = json.loads(path.read_text())
        print(f"{split}: {count_games(payload)} games ({filename})")
        for league, season, game in first_games(payload, limit=3):
            print(f"  - {league} / {season} / {game}")
    print("")

    print("Downloader signatures:")
    print(f"  SoccerNetDownloader: {inspect.signature(SoccerNetDownloader)}")
    print(f"  downloadDataTask: {inspect.signature(SoccerNetDownloader.downloadDataTask)}")
    print("")

    source = Path(inspect.getsourcefile(SoccerNetDownloader)).read_text()
    task_marker = 'elif task == "spotting-ball-2023":'
    task_index = source.find(task_marker)
    if task_index >= 0:
        task_block = source[task_index: task_index + 2200]
        print("spotting-ball-2023 downloader block:")
        print(task_block)
    else:
        print("Could not find spotting-ball-2023 downloader block.")
    print("")

    print("No download was attempted.")
    print(f"If access is approved later, downloader local directory would be: {args.local_directory}")


if __name__ == "__main__":
    main()
