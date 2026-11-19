# napari: browse plates, review segmentation, collect training crops

napari is a window on the user's screen. Start it for them, then tell them
exactly which menu to open and what to click; you can't see the window.

```bash
napari &
```

The widgets are under **Plugins → omero-screen-napari**. They log in to OMERO
with the user's settings from `omero-screen setup`; if a widget can't connect,
run `omero-screen doctor`.

A headless install (on an HPC cluster) has no napari. To add it, reinstall
with `OMERO_SCREEN_HEADLESS=0` (installer option).

---

## Widgets

| Widget | Purpose |
|---|---|
| **WellData** | Load and show the images of a plate well, with masks |
| **Gallery** | A grid of random cell crops, for quick QC |
| **Setup Training** | Create a classifier: classes, crop size, channels |
| **Training** | Label cell crops with the number keys |
| **Classifier Selector** | Pick a classifier; shows crops per class |
| **Session Manager** | Resume, review or delete labelling sessions |
| **Direct Load** | Load crops from OMERO without the gallery step |

## Browse a well

1. **WellData**: enter the plate ID and a well (`C3`), click **Load Well**.
2. Each channel is a layer; toggle and adjust contrast in the layer list.
3. **Show Masks** overlays the nucleus and cell masks from OMERO.

To check segmentation: nuclei not merged or split, cell borders right, border
cells excluded. If it's poor, the fix is a different Cellpose model and a
rerun with `--delete` (see `references/pipeline.md`).

For figure-quality images without napari (galleries, well overviews), use
`omero-screen-images` instead (`references/images.md`).

## Collect training crops

1. **Setup Training**: new classifier, class names, crop size, channels.
2. **Training**: choose the classifier, load an image, press the number key
   for each crop's class.
3. **Session Manager**: resume later; deleting a session warns if another
   session shares its file.

Crops are saved as `.npy` files with a `metadata.json` in the classifier's
folder. Then train with the `cellclass-train` skill.

Two settings users ask about:
- **Isolate the target cell** (`no_background`, on by default): neighbouring
  cells are blanked in each crop. Turn it off only when the context matters
  for the class.
- **Two-channel crops look yellow**: that's red + green; check the channel
  choice in Setup Training.

## The training database

The labels live in a local database, managed with `omero-train`:

```bash
omero-train list                         # classifiers with summary stats
omero-train stats micronuclei_v1         # crops per class
omero-train export micronuclei_v1 --format csv   # --plate/--well to filter
omero-train delete micronuclei_v1        # asks to confirm; ask the user first
```

`omero-train <command> --help` shows the options.

## Troubleshooting

| Problem | Fix |
|---|---|
| No omero-screen-napari in the Plugins menu | Headless install (see above), or napari from another environment: start the `napari` that `install.sh` linked (`command -v napari`) |
| Widget can't connect to OMERO | `omero-screen doctor`; VPN |
| "Keybinding already in use" warning | Harmless when a widget is opened again |
