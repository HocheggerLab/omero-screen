# Deploying Models for Inference

Once you have extracted a TorchScript model you can use it in two ways:

1. **omero-screen pipeline** — automatic inference during plate analysis.
2. **omero-screen-napari** — interactive classification in the Gallery / Classifier Selector widget.

---

## Exported model files

`cellclass-extract best_run.json --save` produces two files:

```
shufflenet2x1_0_c2_l2.pt    # TorchScript model (architecture + weights)
shufflenet2x1_0_c2_l2.json  # Sidecar: channel names, class labels, input size
```

The filename encodes the architecture, number of input channels (`c2`), and number of
class labels (`l2`). Both files must be kept together.

---

## omero-screen pipeline

Set the environment variable before running the pipeline:

```bash
export OMERO_SCREEN_INFERENCE_MODEL=/path/to/shufflenet2x1_0_c2_l2.pt
omero-screen <plate_id>
```

Or pass it directly on the command line:

```bash
omero-screen <plate_id> --inference shufflenet2x1_0_c2_l2.pt
```

Multiple models can be chained (colon-separated in the environment variable, or multiple
`--inference` arguments):

```bash
omero-screen <plate_id> \
    --inference mitosis_model.pt ppase_model.pt
```

Gallery images showing example cells per predicted class are attached as PNG figures to
the OMERO plate. Configure gallery output with:

```bash
export OMERO_SCREEN_INFERENCE_GALLERY_WIDTH=10   # grid size (default: 10)
export OMERO_SCREEN_INFERENCE_BATCH_SIZE=100     # inference batch size (default: 100)
```

See the [pipeline configuration](../configuration.rst) for the full list of variables.

---

## omero-screen-napari

Copy the `.pt` + `.json` pair to the location specified when creating the classifier in
the [Setup Training Widget](../omero-screen-napari/setup_training_widget.md). The
[Classifier Selector](../omero-screen-napari/gallery_widget.md) widget will load the
model and display per-class statistics alongside the database annotation counts.

---

## Verifying a deployed model

Before committing to a production run, evaluate the model on your full dataset:

```bash
cellclass-test -s shufflenet2x1_0_c2_l2.pt ~/data/mitosis-rpe/rois.npz
```

This prints precision, recall, and F1 per class, plus overall accuracy, on the held-out
test split. A model is ready for deployment when test accuracy is ≥ 90 % and no class
has recall below 0.85.
