# Open Data Augmentation

This project should use open soccer data as an accelerator, not as a
replacement for local review. The open datasets are mostly built from
professional broadcast or clean tracking/event feeds, while our target clips are
college video with different cameras, sideline clutter, and local possession
labels.

## Current Target

Use open data to reduce manual work in three places:

- ball-action recognition: pass, cross, shot, free kick, throw-in, goal, out;
- game-state recognition: live play, post-goal stoppage, restart setup, restart
  action;
- active-participant context: player vs referee vs sideline substitute/staff.

Keep Le Moyne and Army reviewed clips as the local truth set.

## Dataset Fit

| Source | Best Use | Direct Fit? | Notes |
| --- | --- | --- | --- |
| SoccerNet Ball Action Spotting | Ball-action/restart priors | High | Broadcast-video task with timestamped ball actions. Strongest candidate for near-term augmentation. |
| SoccerNet Game State Reconstruction | Player/referee/team/field context | High | Best fit for active-participant and field-localization problems, but larger integration. |
| StatsBomb Open Data | Event schema and sequence validation | Medium | Excellent event data, but not a video model source. |
| Metrica sample data | Clean tracking/event logic validation | Medium | Useful for testing downstream possession/event rules with synchronized tracking and events. |

## SoccerNet Ball Action Spotting

Public task page:

- `https://www.soccer-net.org/tasks/ball-action-spotting`

The task is close to our next event-derivation layer. It localizes
ball-related actions in broadcast videos with single timestamps. The listed
2024 classes are:

- `Pass`
- `Drive`
- `Header`
- `High Pass`
- `Out`
- `Cross`
- `Throw In`
- `Shot`
- `Ball Player Block`
- `Player Successful Tackle`
- `Free Kick`
- `Goal`

### Label Crosswalk

| SoccerNet Label | Project Use |
| --- | --- |
| `Pass` | Candidate completed/incomplete pass event; should align with `in_transit` between controlled possessions. |
| `High Pass` | Candidate aerial pass; likely `in_transit`, possible special subtype later. |
| `Cross` | Candidate pass/cross event; useful once field zones are reliable. |
| `Throw In` | Restart action; should not be derived as normal pass from open play. |
| `Free Kick` | Restart action/setup; maps to `restart_setup` or `restart_kick` review layer. |
| `Goal` | Post-goal stoppage/game-state marker. |
| `Out` | Dead-ball or out-of-play marker; useful for ending live possession. |
| `Shot` | Shot event candidate. |
| `Drive` | Possible carry/dribble clue; not a pass. |
| `Header` | Touch subtype; may be pass, clearance, or shot depending context. |
| `Ball Player Block` | Defensive touch/block candidate. |
| `Player Successful Tackle` | Turnover/tackle candidate. |

### Setup Requirements Found So Far

The general SoccerNet Python package can be installed with:

```bash
pip install SoccerNet
```

The package README shows use of `SoccerNet.Downloader.SoccerNetDownloader`.
For Ball Action Spotting, the documented downloader task is:

```python
mySoccerNetDownloader.downloadDataTask(
    task="spotting-ball-2023",
    split=["train", "valid", "test", "challenge"],
    password=<PW_FROM_NDA>,
)
```

That means the full Ball Action Spotting dataset is probably not a one-command
anonymous download. We should expect to request/enter a SoccerNet password or
complete their access flow before using the data.

The SoccerNet task page also says the Ball Action Spotting data has 7 English
Football League videos in 720p and that no extracted features are provided, so
end-to-end methods are expected. This is useful, but heavier than simply
importing JSON labels.

## Practical Integration Options

### Option A: Use SoccerNet Labels As Reference Only

No large install. We use their action taxonomy to harden our schema and review
labels.

Pros:

- immediate;
- no dataset access hurdle;
- helps us avoid inventing odd labels.

Cons:

- does not reduce video-review burden much by itself.

### Option B: Download SoccerNet Ball Action Spotting Data

Install `SoccerNet`, get dataset access/password, download Ball Action Spotting,
and inspect label JSON/video layout.

Pros:

- gives real broadcast-video examples of actions we care about;
- can train/evaluate action spotting later.

Cons:

- may require NDA/password;
- dataset is pro broadcast and may not transfer cleanly to college footage;
- likely heavier than our current scripts.

### Option C: Run/Adapt A Baseline

After dataset access, inspect the official or community baseline and test it on
short clips from Le Moyne/Army.

Pros:

- fastest path to reducing manual event labeling if it works.

Cons:

- likely requires ML dependencies and possibly GPU;
- model output may need post-processing to align with our possession segments.

## Recommended Next Experiment

Do a small, reversible SoccerNet probe:

1. Install only the lightweight SoccerNet package in the local virtual
   environment.
2. Try listing/downloading metadata or a tiny sample for `spotting-ball-2023`.
3. If access is blocked by password/NDA, pause and decide whether it is worth
   requesting access.
4. If access works, inspect one label file and one short video.
5. Build `src/65_map_soccernet_ball_actions.py` to convert SoccerNet actions
   into our candidate schema.
6. Compare the mapped classes against our reviewed Army/Le Moyne moments.

Success criteria:

- We can map SoccerNet labels into our event/game-state labels without weird
  ambiguity.
- We can create a small evaluation table against our reviewed clips.
- We learn something that reduces future manual review or improves event
  derivation.

Stop criteria:

- Access requires a slow approval process and no sample labels are available.
- Baseline code requires a dependency stack that is too heavy for the current
  project phase.
- SoccerNet outputs are too coarse or too domain-shifted for our college clips.

## Sources

- SoccerNet Ball Action Spotting task:
  `https://www.soccer-net.org/tasks/ball-action-spotting`
- SoccerNet package and downloader:
  `https://github.com/SoccerNet/SoccerNet`
- SoccerNet Game State Reconstruction:
  `https://github.com/SoccerNet/sn-gamestate`
- StatsBomb Open Data:
  `https://github.com/statsbomb/open-data`
- Metrica sample data:
  `https://github.com/metrica-sports/sample-data`
