# Possession Lab

A standalone browser tool for producing player-level possession ground truth,
evaluating model predictions, and deriving initial completed-pass candidates.

## Start

From this folder, run:

```sh
python3 -m http.server 8000
```

Then open `http://localhost:8000`.

Use **Load demo** to explore a completed example without opening a video.
Use **Open prepared review** to stream the browser-compatible SoccerTrack
`117093` first-half panorama and load its 15 derived possession segments.

## SoccerTrack v2 workflow

1. Download a SoccerTrack v2 BAS annotation JSON file.
2. Use **Import SoccerTrack actions**.
3. Select the half and open its matching panorama video.
4. Inspect the standardized action timeline.
5. Use **Derive possession** to create conservative, review-required possession
   segments from identifiable player actions.
6. Export the standardized actions or reviewed possession labels.

The current derivation treats passes, drives, headers, high passes, crosses,
shots, throw-ins, free kicks, and successful tackles as evidence that the
identified actor controls or has just controlled the ball. It does not infer
control from blocks, outs, or goals.

Possession Lab normalizes both documented half-relative BAS timestamps and
current Drive files that use a global match timeline. The common action schema
preserves `half`, seconds within the selected half video, source time, the
25 fps frame index, team, player, visibility, original label, and source.
Possession Lab reviews and exports one selected half at a time.

SoccerTrack v2 data is distributed under CC BY 4.0. Preserve attribution when
using or redistributing derived data:

> SoccerTrack v2, Atom Scott et al., CC BY 4.0.

## Labeling workflow

1. Open a short video clip.
2. Seek to the start of a possession segment and press `I`.
3. Seek to its end and press `O`.
4. Choose the possession state. For `controlled`, enter the team and player ID.
5. Press `Enter` to add the segment.
6. Export labels when the clip is complete.

Useful shortcuts:

- `Space`: play or pause
- `Left` / `Right`: move 0.1 seconds
- `Shift` + `Left` / `Right`: move 1 second
- `I`: mark segment start
- `O`: mark segment end
- `Enter`: add segment

## Initial labeling protocol

- Label 3-5 clips of 20-60 seconds each.
- Include ordinary possession, contested balls, turnovers, airborne balls, and
  out-of-play periods.
- Use stable tracking IDs when available. Temporary IDs are fine.
- Mark `controlled` only when one player clearly controls the ball.
- Use `contested`, `loose`, `airborne`, `out_of_play`, or `unknown` instead of
  forcing a player assignment.
- Do not label a brief touch as controlled possession unless the player
  establishes control.

## Evaluation

Import a prediction JSON using the same schema as exported labels. The tool
calculates time-weighted state, team, and player accuracy across labeled time.

## Completed pass rule

The first-pass detector reports a completed pass when:

- two consecutive `controlled` segments belong to the same team,
- the controlling player changes, and
- the transition time is between zero and the configured maximum.

This deliberately produces candidates. Later versions should distinguish
intentional passes from deflections, touches, and handovers using ball motion
and surrounding-player context.

## Label JSON schema

```json
{
  "schema_version": "1.0",
  "clip": "example.mp4",
  "segments": [
    {
      "id": "uuid",
      "start": 2.4,
      "end": 4.9,
      "state": "controlled",
      "team": "home",
      "player": "home_07",
      "confidence": 1,
      "notes": ""
    }
  ]
}
```
