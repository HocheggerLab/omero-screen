# omero-train CLI Reference

## What this tool does

`omero-train` is a command-line tool that lets you manage your training database, inspect annotation statistics, and export data — without opening Napari. It is useful for:

- Quickly checking how many cells have been labelled for a classifier.
- Exporting annotation data to CSV for downstream analysis or sharing with collaborators.
- Cleaning up classifiers or sessions that are no longer needed.
- Migrating training data created before the database was introduced.

## Installation

`omero-train` is installed automatically as part of `omero-screen-napari`. After installation, it is available in your terminal:

```bash
omero-train --help
```

## Commands overview

| Command | What it does |
|---------|-------------|
| `list` | Show all classifiers with summary statistics |
| `stats` | Show detailed statistics for one classifier |
| `export` | Export annotations to CSV, JSON, or Parquet |
| `delete` | Remove a classifier and its sessions |
| `migrate` | Import existing NPY files into the database (for legacy data) |

---

## `omero-train list`

Lists all classifiers in the training database with a summary table.

```bash
omero-train list
```

**Output columns:**

| Column | Description |
|--------|-------------|
| ID | Internal database ID |
| Name | Classifier name |
| Sessions | Number of annotation sessions |
| Annotations | Total crops labelled |
| Classes | Class names defined for this classifier |
| Created | Date the classifier was first created |

**Example output:**
```
┌────┬─────────────┬──────────┬─────────────┬──────────────────────────────────┐
│ ID │ Name        │ Sessions │ Annotations │ Classes                          │
├────┼─────────────┼──────────┼─────────────┼──────────────────────────────────┤
│  1 │ mitosis-rpe │        4 │         320 │ interphase, mitosis, apoptosis   │
│  2 │ drug-u2os   │        2 │         150 │ normal, stressed                 │
└────┴─────────────┴──────────┴─────────────┴──────────────────────────────────┘
```

---

## `omero-train stats`

Shows detailed statistics for a single classifier, identified by name or database ID.

```bash
omero-train stats <name-or-id>
```

**Examples:**
```bash
omero-train stats mitosis-rpe
omero-train stats 1
```

**Output includes:**
- General info (ID, total sessions, total annotations, classes)
- Class distribution table with counts and percentages
- Per-session breakdown (plate, well, images, timepoint, cell cycle filter, cell count, class distribution)

**Example output:**
```
Classifier: mitosis-rpe
Sessions: 4  |  Total cells: 320  |  Classes: interphase, mitosis, apoptosis

Class Distribution
┌─────────────┬───────┬────────────┐
│ Class       │ Count │ Percentage │
├─────────────┼───────┼────────────┤
│ interphase  │   224 │      70.0% │
│ mitosis     │    80 │      25.0% │
│ apoptosis   │    16 │       5.0% │
│ unassigned  │     0 │       0.0% │
└─────────────┴───────┴────────────┘

Sessions
┌────┬──────────┬──────┬────────┬──────────────┬────────┬───────────────┐
│ ID │ Plate ID │ Well │ Images │ Cell Cycle   │ Cells  │ Distribution  │
├────┼──────────┼──────┼────────┼──────────────┼────────┼───────────────┤
│  1 │     3869 │ E2   │ All    │ All          │     80 │ interph.: 60  │
│  2 │     3869 │ F3   │ 0, 1   │ All          │     80 │ mitosis: 20   │
└────┴──────────┴──────┴────────┴──────────────┴────────┴───────────────┘
```

---

## `omero-train export`

Exports all annotations for a classifier to a file. Useful for sharing data, computing statistics in R or Python, or backing up before deletion.

```bash
omero-train export <name-or-id> [options]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--format` | `csv` | Output format: `csv`, `json`, or `parquet` |
| `--output` | auto | Output file path. Defaults to `<classifier-name>_annotations.<format>` in the current directory |
| `--plate` | all | Export only sessions from a specific plate ID |
| `--well` | all | Export only sessions from a specific well |

**Examples:**

```bash
# Export everything to CSV
omero-train export mitosis-rpe

# Export to JSON with a custom filename
omero-train export mitosis-rpe --format json --output ~/Desktop/mitosis_data.json

# Export only data from plate 3869
omero-train export mitosis-rpe --plate 3869

# Export only well A1 from plate 3869
omero-train export mitosis-rpe --plate 3869 --well A1
```

**Output columns (CSV):**

| Column | Description |
|--------|-------------|
| session_id | Database ID of the session |
| plate_id | OMERO plate ID |
| well | Well position |
| image_id | OMERO image ID |
| timepoint | Timepoint index |
| cell_index | Index of the crop within the session |
| class_label | Assigned class name |
| centroid_row | Row coordinate of the cell centroid in the full image |
| centroid_col | Column coordinate of the cell centroid |

---

## `omero-train delete`

Removes one or more classifiers and all their associated sessions and annotation records. This also deletes the NPY data files from disk.

```bash
omero-train delete <name-or-id> [<name-or-id> ...] [options]
```

**Options:**

| Option | Description |
|--------|-------------|
| `--yes` | Skip the confirmation prompt (useful in scripts) |
| `--plate <id>` | Delete only sessions from a specific plate, leaving the rest of the classifier intact |

**Examples:**

```bash
# Delete a single classifier (will ask for confirmation)
omero-train delete mitosis-rpe

# Delete without confirmation prompt
omero-train delete mitosis-rpe --yes

# Delete only the sessions from plate 3869, keep the classifier and other sessions
omero-train delete mitosis-rpe --plate 3869

# Delete multiple classifiers at once
omero-train delete mitosis-rpe drug-u2os --yes
```

> ⚠ **Deletion is permanent.** Export your data first with `omero-train export` if you may need it later.

---

## `omero-train migrate`

Imports existing NPY training data files (created before the database was introduced) into the training database. This is a one-time operation for legacy data.

```bash
omero-train migrate [options]
```

**Options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--path` | `~/omeroscreen_trainingdata` | Root directory to scan for NPY files |
| `--dry-run` | off | Print what would be imported without actually writing to the database |

**Examples:**

```bash
# Preview what would be migrated without making any changes
omero-train migrate --dry-run

# Run the migration from the default location
omero-train migrate

# Run from a custom directory
omero-train migrate --path /data/training_backup
```

The command scans the directory tree for `.npy` files and `metadata.json` files, infers the classifier structure from the folder layout, and creates the corresponding database records.

---

## Where is the database?

The training database is an SQLite file stored at:

```
~/omeroscreen_trainingdata/training_data.db
```

You can open this file with any SQLite browser (e.g. [DB Browser for SQLite](https://sqlitebrowser.org/)) if you want to query the data directly.

---

## Tips

- Run `omero-train list` regularly to get a quick health check on your training data.
- Before deleting anything, always run `omero-train export` first.
- Use `omero-train stats` to check class balance before starting a training run — heavily imbalanced classes (e.g. 95% interphase, 5% mitosis) may need more data for the rare class.
- The `--plate` filter on both `export` and `delete` is useful when one plate's data turned out to be low quality and you want to remove it without losing everything else.
