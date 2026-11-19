# Classifiers: collect crops, then publish and apply

A cell classifier (micronuclei, mitotic, …) goes through four stages:

```
1. collect labelled crops in napari            ← this file
2. build the dataset, train, evaluate, extract ← the cellclass-train skill
3. publish the model to OMERO                  ← this file
4. classify plates: omero-screen --inference   ← this file
```

**For stage 2, load the `cellclass-train` skill.** It has the standard sweep,
the evaluation report and the extraction; don't train from this file.

---

## 1. Collect labelled crops in napari

1. Start napari for the user (`napari &`) and tell them where to click.
2. **Setup Training** widget: create a classifier, name the classes, set the
   crop size.
3. **Training** widget: choose the classifier, load a plate/well, label crops
   with the number keys.
4. **Session Manager**: resume or review earlier sessions.

Each session saves `.npy` files plus a `metadata.json` (channels, crop size)
into the classifier's training-data folder. Aim for at least 200 crops per
class, and similar numbers per class if you can.

Details of the widgets: `references/napari.md`.

## 3. Publish the model to OMERO

`cellclass extract` (stage 2) writes `<name>.pt` and a `<name>.json` sidecar
(labels, channels, scores). Upload both, after asking the user:

```bash
omero-screen models publish micronuclei_c2_l2.pt            # finds the .json beside it
omero-screen models publish micronuclei_c2_l2.pt --replace  # update an existing one
```

They go to the user's `Classifiers` project in OMERO, so the pipeline can find
them on any machine.

## 4. Classify plates

```bash
omero-screen 1821 --inference micronuclei_c2_l2             # name, without .pt
omero-screen 1821 1822 --inference micronuclei mitotic --gallery 15
```

Ask before running (it writes to OMERO). Each classifier adds a column to the
plate's results and attaches an example gallery per class. Plot the results
with `classification_plot` (`references/plotting.md`).

## Troubleshooting

| Problem | Fix |
|---|---|
| `--inference NAME` not found | Publish it first (`omero-screen models publish`), as the same OMERO user |
| Classes look wrong on a plate | Check that the plate's channels match the `channels` in the model's `.json` sidecar |
| Too few crops of one class | Collect more in napari before training; class balance matters more than total count |
