#!/bin/bash

# Echo a command then run it
exe() {
  echo "\$ $@" ;
  "$@" ;
  if [ $? != 0 ]; then
    banner "fail"
    exit $?
  fi
}

# Check OMERO connection and authenticate
check_omero_connection() {
  # Try to list sessions to check connection
  if omero sessions list &>/dev/null; then
    echo "Found active OMERO session. Closing..."
    omero sessions keepalive none
    omero logout
  fi

  if [ ${SERVER_FLAG} ]; then
    read -p "OMERO host [localhost]: " host
    host=${host:-localhost}
    read -p "OMERO port [4064]: " port
    port=${port:-4064}
    read -p "OMERO username [root]: " username
    username=${username:-root}

    if ! omero login ${username}@${host} -p ${port}; then
      echo "ERROR: Failed to connect to OMERO server"
      exit 1
    fi
  else
    if ! omero login root@localhost -p 4064; then
      echo "ERROR: Failed to connect to OMERO server"
      exit 1
    fi
  fi
  echo "Successfully connected to OMERO server"
}

# Print usage
usage() {
  echo "$0 [OPTION]..."
  echo "Imports test data into an OMERO server."
  echo "Requires:"
  echo " - omero-py environment"
  echo " - Running OMERO server"
  echo "Options:"
  echo "-x     Execute import"
  echo "-d=s   Screen data directory (containing Operetta screens with */Images/Index.idx.xml)"
  echo "-s     Specify custom server details (default: root@localhost -p 4064)"
  exit 0
}

if [ $# = 0 ]; then
  usage
fi

DIR='.'

# Parse options
while getopts ":xd:s" opt; do
  case $opt in
    x)
      EXECUTE=1
      ;;
    d)
        DIR=$OPTARG
        ;;
    s)
        SERVER_FLAG=1
        ;;
    \?)
      echo "Invalid option -$OPTARG" >&2
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

# Check for bad params
shift $((OPTIND-1))
if [ "$*" ]; then
  echo "ERROR - Unused parameters:$@:"
  echo
  usage
fi

if [ ! -d $DIR ]; then
  echo "ERROR - Missing directory: $DIR"
  exit 1
fi

if [ ! ${EXECUTE} ]; then
  exit 0
fi

# Check OMERO connection before proceeding
check_omero_connection

# Create the project for the screen processing images, e.g. flat-field correction
exe omero obj new Project name="Screens"

# Import the screens
shopt -s nullglob
for s in "${DIR}"/*/Images/Index.idx.xml; do
    # Add debug output to see what paths we're working with
    echo "Processing file: $s"

    # Get the parent directory of "Images" folder using absolute paths
    screen_dir=$(cd "$(dirname "$(dirname "$s")")" && pwd)
    plate_name=$(basename "$screen_dir")

    echo "Screen directory: $screen_dir"
    echo "Plate name: $plate_name"

    # Make sure to quote all variables to handle spaces and special characters
    exe omero import "${s}" -n "${plate_name}"
done
