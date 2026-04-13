#!/bin/bash -l
#SBATCH -J os3604.2016550
#SBATCH -o os3604.2016550."%j".out
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
  uv run python send-mail.py -m "Torch GPU unavailable for os3604.2016550.sh" -s "os3604.2016550.sh failed" hh65@sussex.ac.uk
  exit $code
fi
set -e
runcmd uv run python run_omero_screen.py 3604 --inference nuclei4
msg Sending result e-mail using send-mail.py
python send-mail.py -m '
          Job results: os3604.2016550
          Plate: 3604
          ' -s 'Job results: os3604.2016550' hh65@sussex.ac.uk
msg Done
rm os3604.2016550.sh
