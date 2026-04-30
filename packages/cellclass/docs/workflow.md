# Cell Classification

Classify cell data using neural networks.

## Set-up

Prepare the environment using [uv](https://docs.astral.sh/uv/):

    uv sync

Programs located in `src/bin` can now be executed using `uv run`. To omit the `uv run`
prefix activate the environment using:

    source .venv/bin/activate

## Data preparation

Cell classification examples are pickled numpy data files saved as a dictionary.
Each example has an image in (YXC) format, a mask for the region of interest (YX)
and a classification label (str).
The `data` key is a tuple of lists of the images and masks. The `target`
key is the label for each training example.

    {
      'data': ([img1, img2, ..., imgN], [mask1, mask2, ..., maskN]),
      'target': [label1, label2, ..., labelN]
    }

Multiple data files may be present in a dataset. The neural network should be trained on preprocessed images.
These will be preprocessed to maintain only the data of relevance to the classification problem.
A typical workflow is to extract the (1, 99) percentile of the image data, zero all pixels outside the mask
and convert to unsigned 8-bit format (for memory efficiency). The data can be saved into one numpy file
to allow fast loading into memory. For example:

    uv run ./create_dataset.py ~/images/training_data --channels DAPI

Channel names must be specified explicitly using the `--channels` flag, with one name per channel in the
order they appear in the image data. For example, a two-channel DAPI/RFP dataset:

    uv run ./create_dataset.py ~/images/training_data --channels DAPI RFP

Alternatively, channel names can be defined in a `metadata.json` file in the data directory (see below),
in which case the `--channels` flag can be omitted.

This will create a single `.npz` file in the data directory named `rois.npz` containing the preprocessed images.
This program has different options; see the help for details.

Examples of each class can be extracted from the dataset using:

    ./sample_images.py ~/images/training_data/rois.npz --output ~/images/training_data

This will create a tif file for each class containing a random selection of images. The tif file has been written
using `ImageJ` hyperstack format. Multiple channel images can be viewed in `ImageJ` using the channels tool
(`Image > Color > Channels Tool...`) and selecting `Composite` .

## Training

### Weights & Biases

It is recommended to log training runs to [Weights & Biases](https://wandb.ai/). The environment can be configured
for a user with the following command:

    wandb login

Note that the `wandb` command will be available in the `uv` environment.
Visit the url https://wandb.ai/authorize and login. This will generate a key to provide to the login command.
The environment will now be configured to automatically login to Weights & Biases.

### Training

The training can be performed using multiple options and models.
Ideally training should be performed on an accelerator device such as a GPU.
To train a model on an Apple MPS device and log to Weights & Biases:

    ./run_training.py ~/images/training_data/rois.npz -d mps --model efficientnetb3s --wandb -n test -s run1

This program has different options; see the help for details. By default the input data is split into
non-testing and testing. The non-testing data is then split into training and validation. The training
is performed on the training data and validated on the validation data. The testing data is separate and
never used to influence training (e.g. determine model convergence). It is used to provide the expected
model performance on new data.

Training will create output files using the provided prefixes. In this example it is `test` for the checkpoint
files and `run1` for the metadata file.

The checkpoint name is used to save the training state.
This allows the run to be restarted from the current state, for example in the event that training crashes
due to a fixable issue it can be restarted from the previous point. Note that some training objects
may not be supported and are created again, for example the convergence checker. Consider restarts
as a work-in-progress beta feature.

The best model is saved into a checkpoint file with the `.best` suffix. This is the best performing model
on the validation data. It should be used to extract the final model weights and used for evaluation.

`.json`: Metadata file containing details of all the arguments used to run training.
Training can be restarted by passing this file to the program:

    ./run_training.py test.json

Note: Using the `--wandb` flag will log to Weights & Biases. At the end of training the metadata and checkpoint
files are uploaded as artifacts of the training run.

### Batch Training

Multiple different options for training can be prepared into a training batch file.
These are arguments to pass to the `run_training.py` program, for example:

    /path/to/rois.npz --model densenet161 --lr-scheduler plateau --no-loss-weights --batch-size 32
    /path/to/rois.npz --model densenet161 --lr-scheduler plateau --no-loss-weights --batch-size 64
    /path/to/rois.npz --model densenet161 --lr-scheduler plateau --no-loss-weights --batch-size 128
    /path/to/rois.npz --model densenet201 --lr-scheduler plateau --no-loss-weights --batch-size 32
    /path/to/rois.npz --model densenet201 --lr-scheduler plateau --no-loss-weights --batch-size 64
    /path/to/rois.npz --model densenet201 --lr-scheduler plateau --no-loss-weights --batch-size 128

These can be processed into a script:

    ./batch_training.py train.txt --script batch.sh

This will create a batch script to execute the training program for each line in the batch file. This processing script
ignores empty lines and commented lines beginning with `#`. This allows the same batch file to be used repeatedly
to track all the training runs that have been performed on a dataset. Note that old runs should be commented out
before rebuilding the training script.

The `batch_training.py` program will create appropriate names and output filenames to avoid overwriting existing files.
It will log results to Weights & Biases and generate a W&B run name using the concatenated arguments.

By default a batch script is created allowing the command to be verified before executing. The script can also
run each training run in a background process. This is not recommended unless the number of runs is small
as all will run concurrently. This option is to be used on a high specification machine that may have multiple
graphics cards. Each run can have the device identified as one of the arguments, e.g. `--device cuda:0`.

### Training on a SLURM cluster

#### Artemis

The University of Sussex Artemis cluster can be used to perform multiple GPU based training jobs.
Documentation on Artemis is here: https://artemis-docs.hpc.sussex.ac.uk/artemis/.
This requires VPN access.

#### Batching SLURM Jobs

The program `sbatch_training.py` will create and execute a script to submit a training job to a SLURM cluster
using `sbatch`. The script accepts some parameters to configure options; see the help for details. However
any parameters after the `--args` argument are passed to the `run_training.py` program.

To run a single job on a SLURM cluster the program is run as per `run_training.py`:

    ./sbatch_training.py --args ~/images/training_data/rois.npz -d mps --model efficientnetb3s --wandb -n test -s run1

To create a batch of jobs to submit to a SLURM cluster the program name can be passed to the batching script:

    ./batch_training.py train.txt --script batch.sh --cmd "./sbatch_training.py --args"

This effectively creates the same `batch.sh` script as previously but will run `sbatch_training.py` for
each run rather than `run_training.py`.

## Extracting Models

The training data metadata can be used to extract details of the model from the checkpoint files
such as the performance metrics, or save the model weights and metadata:

    ./extract_model.py training.json
    ./extract_model.py training.json --save

The `save` option will create a file name using the model name, the number of input channels
in the images and the number of class labels. The scripted model architecture and weights
are saved to a `.pt` file and a separate `.json` metadata file contains the channel names and class labels.
The `.pt` file uses the [TorchScript](https://pytorch.org/docs/stable/jit.html) format.

## Testing Models

Test a model from the checkpoint file on a dataset; requires knowing the model class:

    ./test_model.py --model densenet201 -n run123.pt.best ~/images/training_data/rois.npz

Test a TorchScript model on a dataset; requires the `.json` sidecar metadata file describing
the architecture (channels and classes, and input image size):

    ./test_model.py -s densenet201_c1_l3.pt ~/images/training_data/rois.npz

This script will create a performance metrics report on the precision, recall and accuracy of the model.
