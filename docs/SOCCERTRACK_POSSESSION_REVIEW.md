# SoccerTrack Possession Review

## Required Local Files

For one match ID, place these files under `data/soccertrack/<match_id>/`:

```text
bas/<match_id>_12_class_events.json
videos/<match_id>_panorama_1st_half.mp4
videos/<match_id>_panorama_2nd_half.mp4
```

The Hugging Face link is currently unavailable, but the official public Google
Drive mirror works. Match data is excluded from Git.

The current Drive BAS files use global match timestamps even though the
published format describes half-relative timestamps. Possession Lab normalizes
both forms to the matching half video.

## Start Possession Lab

```sh
cd tools/possession-lab
python3 -m http.server 8000
```

Open `http://localhost:8000`.

## First Review Pass

The first review batch is prepared from match `117093`:

```text
data/soccertrack/117093/bas/117093_1st_half_review_0000_0060_actions.json
data/soccertrack/117093/videos/117093_panorama_1st_half_review_0000_0060.mp4
```

It covers the first 60 seconds of the first half and contains 17 actions that
produce 15 conservative possession segments.

1. Import the match BAS JSON with **Import SoccerTrack actions**.
2. Select **First half**.
3. Open the matching first-half panorama video.
4. Click **Derive possession**.
5. Review an initial representative 20-60 second section.
6. Correct player, team, timing, and possession state.
7. Export the half-specific labels.

Repeat with the second half after the first-half workflow is reliable.

Match `117092` was screened and rejected for this first review because its BAS
file contains corrupted first-half timestamps extending to 135 minutes.

For each segment, verify:

- player and team;
- start and end time;
- `controlled`, `contested`, `loose`, `airborne`, `out_of_play`, or `unknown`;
- whether any resulting teammate-to-teammate transition is a genuine pass.

Do not force ambiguous moments into controlled possession.
