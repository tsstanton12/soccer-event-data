from pathlib import Path
import shutil
import argparse


BASE_DIR = Path("data/harvested_frames/patriot_league")


VENUE_TARGETS = {
    "colgate": {
        "tracking_failures": None,
        "negative_frames": None,
        "tiny_ball": 150,
        "low_confidence": 100,
        "review_all": 25,
    },
    "loyola": {
        "tracking_failures": None,
        "negative_frames": 300,
        "tiny_ball": 150,
        "low_confidence": 500,
        "review_all": 50,
    },
    "lehigh": {
        "tracking_failures": None,
        "negative_frames": 300,
        "tiny_ball": 150,
        "low_confidence": 500,
        "review_all": 50,
    },
    "bucknell": {
        "tracking_failures": None,
        "negative_frames": 250,
        "tiny_ball": 200,
        "low_confidence": 500,
        "review_all": 50,
    },
    "navy": {
        "tracking_failures": None,
        "negative_frames": 200,
        "tiny_ball": 175,
        "low_confidence": 400,
        "review_all": 50,
    },
    "lafayette": {
        "tracking_failures": None,
        "negative_frames": None,
        "tiny_ball": 400,
        "low_confidence": 300,
        "review_all": 50,
    },
    "army": {
        "tracking_failures": None,
        "negative_frames": None,
        "tiny_ball": 500,
        "low_confidence": 300,
        "review_all": 50,
    },
    "holy_cross": {
        "tracking_failures": None,
        "negative_frames": None,
        "tiny_ball": 500,
        "low_confidence": 350,
        "review_all": 50,
    },
    "american": {
        "tracking_failures": None,
        "negative_frames": None,
        "tiny_ball": 500,
        "low_confidence": 250,
        "review_all": 50,
    },
    "bu": {
        "tracking_failures": None,
        "negative_frames": None,
        "tiny_ball": 500,
        "low_confidence": 350,
        "review_all": 50,
    },
}


def evenly_select(files, target):
    files = sorted(files)

    if target is None or len(files) <= target:
        return files

    return [
        files[round(i * (len(files) - 1) / (target - 1))]
        for i in range(target)
    ]


def build_upload_package(venue_name):
    venue_name = venue_name.lower()
    venue_dir = BASE_DIR / venue_name

    if venue_name not in VENUE_TARGETS:
        print(f"Unknown venue: {venue_name}")
        print(f"Known venues: {', '.join(VENUE_TARGETS.keys())}")
        return

    if not venue_dir.exists():
        print(f"Venue folder not found: {venue_dir}")
        return

    upload_dir = venue_dir / "upload_package"

    if upload_dir.exists():
        shutil.rmtree(upload_dir)

    upload_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nBuilding upload package for: {venue_name}")
    print(f"Source: {venue_dir}")
    print(f"Output: {upload_dir}")
    print("-" * 50)

    total = 0
    targets = VENUE_TARGETS[venue_name]

    for category, target in targets.items():
        src = venue_dir / category

        if not src.exists():
            print(f"{category}: missing folder")
            continue

        files = sorted(src.glob("*.jpg"))
        selected = evenly_select(files, target)

        for f in selected:
            new_name = f"{venue_name}_{category}_{f.name}"
            shutil.copy2(f, upload_dir / new_name)

        print(
            f"{category}: copied {len(selected)} "
            f"from {len(files)} available "
            f"(target: {'all' if target is None else target})"
        )

        total += len(selected)

    print("-" * 50)
    print(f"TOTAL COPIED FOR {venue_name.upper()}: {total}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--venue",
        help="Venue name, e.g. loyola, lehigh, bucknell, or all",
        required=True,
    )

    args = parser.parse_args()

    if args.venue.lower() == "all":
        for venue in VENUE_TARGETS:
            build_upload_package(venue)
    else:
        build_upload_package(args.venue)


if __name__ == "__main__":
    main()