---
name: track-review
description: Co-review tracked cells with a human in napari — set up a review queue, drive the Track Review widget through the omero-screen napari-mcp tools, inspect cells with filmstrips and traces, and propose lineage or mask edits for the reviewer to confirm. Use when the user wants to check, curate or review tracks, cell fates or a review queue for a timelapse plate.
---

# Track review with a human (omero-screen)

You and the reviewer check tracked cells together. **You navigate, measure and
propose; the reviewer decides.** You never change data directly: the only tool
that changes anything is `propose_edit`, and a proposal takes effect only when
the reviewer confirms it in the Track Review widget.

## Concepts

- **Plate data:** CellView measurements with raw Trackastra tracks
  (`track_id_raw`). The automatic FUCCI-gated repair (geminin drop = real
  mitosis) runs on these every time; it is never edited.
- **Anchor / item id:** a cell is named by the raw label it has in the start
  frame, e.g. `C2-t72-L524`, which is label 524 at frame 72 in well C2.
- **Edit log** (`edits.jsonl` beside `queue.json`): every correction (link,
  unlink, absorb, drop, events, exclude, masks). Curated data = repair +
  replay of the log. Entries record author (`human` / `agent`) and who confirmed.
- **Verdicts** (`decisions.json`): the reviewer's accept / correct / reject /
  unsure per item.
- **Window:** cells present at frame 72 (24 h) are followed to frame 168
  (56 h). Outcomes: divided, mitotic_exit_no_division, mitotic_arrest, death,
  no_mitosis, lost (censored). Excluded: debris, stationary, no_reporter.

## Setup (the reviewer runs these; check before starting)

1. Every well to review must be in the zarr cache (Welldata widget → Plate Info → Cache
   Plate). `review_goto` fails for an
   uncached well.
2. Build a queue (random sample per well, flagged cells + audit fraction):
   ```
   cellview --db ~/.cellview/cellview.duckdb review-queue 5054 \
       --well C2 --well C3 --well C4 --sample 50 --seed 1 \
       --log <dir>/edits.jsonl --out <dir>
   ```
   For the paper, `<dir>` is `omero-screen-paper/05_tracking/raw/review`.
3. napari with the agent hook (the fork until upstream releases it):
   ```
   cd ~/code/omero-screen
   OMERO_SCREEN_REVIEW_QUEUE=<dir>/queue.json \
   uv run --with "napari-mcp @ git+https://github.com/HocheggerLab/napari-mcp@plugin-tools" napari
   ```
   Open *Plugins → Track Review Widget*, then *napari-mcp → Start*.
4. Register once: `claude mcp add --transport http napari http://127.0.0.1:9999/mcp`
   (tools appear in the next agent session).

## Tools

| Tool | Use |
|---|---|
| `review_state` | Progress, the cell on screen, current frame, pending proposals |
| `review_list(well, flag, status)` | Pick what to look at next |
| `review_goto(item_id, frame)` | Put a cell on the reviewer's screen |
| `filmstrip(item_id, start, stop)` | **Look first.** Crops following the cell, with trace and phase bands |
| `cell_trace(item_id)` | Numbers per frame: label, gap, area, PIP, geminin, phase |
| `continuation_candidates(item_id, frame)` | Ranked continuations at a break |
| `label_at(well, frame, y, x)` | Which nucleus is at a position |
| `propose_edit(item_id, op, args, reason)` | Suggest a change (see below) |
| `proposal_status`, `cell_edits`, `validate`, `session_summary` | Follow-up and reporting |

napari-mcp's own `screenshot` shows the live view. It cannot follow a moving
cell, so use `filmstrip` for history. Never use `install_packages`. Use
`execute_code` only for debugging, and say so.

## Per-cell protocol

1. `review_goto`, then `filmstrip` for the window, then `cell_trace` if
   numbers matter.
2. Read the flag (`reason`) and answer its question:
   - **lost**: where does the cell go after the last frame? Check
     `continuation_candidates` at the break. A good continuation is close
     (RPE-1 moves ~0.5 diameters per frame, occasionally 2–3), has similar area,
     and has continuous PIP/geminin. Propose `link` only if one candidate is
     clearly best; otherwise ask.
   - **gap / link_ambiguous**: same cell on both sides? Compare reporters
     before and after.
   - **fragment**: one nucleus in pieces (keep) or two cells (`drop` the
     neighbour's piece)?
   - **phase_order**: phases running backwards usually means a wrong link or
     a floating/dying cell over the nucleus. Find the frame.
   - **outcome_review**: confirm death / slippage / arrest from the filmstrip.
     Death: shrinking, condensing, fragmenting nucleus. Slippage: geminin
     drops without two daughters.
   - **edge, reporter_unclear, refractory, audit**: check the whole track.
3. Biology to remember:
   - Early S has **both PIP and geminin low**, briefly. That is not "no reporter".
   - Stationary objects are debris.
   - Dying cells float over neighbours and blend in and out of their masks.
   - A real mitosis has geminin dropping in both daughters.
4. Propose with a reason the reviewer can check on screen, e.g. *"link at
   t156 to candidate 1 (label 3132): 2.4 diameters, area 0.74×, PIP and
   geminin continuous"*. One proposal per decision. Wait for
   `proposal_status` before proposing something that depends on it.
5. The **verdict is the reviewer's**. Suggest one, but they click it.

## Proposable operations

| op | args |
|---|---|
| `link` | `{"frame": t, "label": L}`: from frame t the cell is label L |
| `unlink` | `{"frame": t}` |
| `absorb` / `drop` | `{"frames": [t0, t1], "label": L}` |
| `event` | `{"kind": "mitosis" \| "death" \| "slippage", "frame": t}` |
| `set_outcome` | `{"outcome": "..."}` |
| `exclude` | `{"reason": "debris" \| "no_reporter" \| ...}` |
| `note` | `{"text": "..."}` |

Mask splits and drawing are the reviewer's (*Split nucleus*, *Draw nucleus*).
Suggest them in words.

## Defer to the reviewer when

- two candidates are similarly plausible;
- the call is biological (is this death? is this one cell?);
- the image is too poor to see;
- a proposal was rejected: read its note and don't repeat it.

## End of a session

1. `session_summary`: report reviewed per well, verdicts, edits by author,
   and agreement (confirmed / resolved proposals).
2. Write a short progress-log entry (vault `&OmeroScreenTracking_progresslog`)
   with the counts and any recurring problem types.
3. Rebuild the analysis tables if asked:
   `uv run python 05_tracking/build/build_cell_fates.py` (paper repo).
