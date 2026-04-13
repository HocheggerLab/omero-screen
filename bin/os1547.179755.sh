#!/bin/bash -l
#SBATCH -J os1547.179755
#SBATCH -o os1547.179755."%j".out
#SBATCH -p gpu
#SBATCH --mail-user hh65@sussex.ac.uk
#SBATCH --mail-type=END,FAIL
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu
        
function msg {
  echo $(date "+[%F %T]") $@
}
function runcmd {
  msg exec: $@
  $@
}
set -e
export PYTHONPATH=$(cd ../ && pwd)
msg PYTHONPATH=$PYTHONPATH
set +e
runcmd uv run python torch-test.py
code=$?
if [ $code -ne 0 ]; then
  msg Torch test exit code: $code
  uv run python send-mail.py -m "Torch GPU unavailable for os1547.179755.sh" -s "os1547.179755.sh failed" hh65@sussex.ac.uk
  exit $code
fi
set -e
runcmd uv run python run_omero_screen.py 1547 --inference ppase_screen2
msg Sending result e-mail using send-mail.py
python send-mail.py -m '
          Job results: os1547.179755
          Plate: 1547
          ' -s 'Job results: os1547.179755' hh65@sussex.ac.uk
msg Done
rm os1547.179755.sh
