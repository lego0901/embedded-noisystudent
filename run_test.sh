#!/bin/bash

# If you want to excludes some tests, then use this script
# NOT_ANALYZE="--not_analyze_top5=True --not_analyze_p=True"

# Note that CIFAR-10-C and CIFAR-10-P are quite big data, which takes a long
# time to download the data.
# If you want to omit those tests, please use "--not_analyze_c=True" or
# "--not_analyze_p=True" options on NOT_ANALYZE. More options are available in
# test.py file.
NOT_ANALYZE=""
OPT="--batch_size_test=512 --batch_size_fgsm=256 --num_workers=4 --device=cuda"


echo "================================================================"
echo "                       downloading test data                    "
echo "================================================================"

BASE_DIR=$(dirname $0)
DATA_DIR="$BASE_DIR/data"

cd $BASE_DIR

if [[ ! -e $DATA_DIR ]]; then
  mkdir $DATA_DIR
fi

# download and extract CIFAR-10-C data
if [[ ! $NOT_ANALYZE == *"--not_analyze_c=True"* ]]; then
  cd $DATA_DIR

  if [[ ! -e CIFAR-10-C ]]; then
    if [[ ! -e CIFAR-10-C.tar ]]; then
      echo "Downloading CIFAR-10-C test data..."
      curl -O https://zenodo.org/record/2535967/files/CIFAR-10-C.tar?download=1
    fi

    echo "Extracting CIFAR-10-C test data..."
    tar -xvf CIFAR-10-C.tar -C "."
  fi

  cd ..
fi

# download and extract CIFAR-10-P data
if [[ ! $NOT_ANALYZE == *"--not_analyze_p=True"* ]]; then
  cd $DATA_DIR

  if [[ ! -e CIFAR-10-P ]]; then
    if [[ ! -e CIFAR-10-P.tar ]]; then
      echo "Downloading CIFAR-10-P test data..."
      curl -O https://zenodo.org/record/2535967/files/CIFAR-10-P.tar?download=1
    fi

    echo "Extracting CIFAR-10-P test data..."
    tar -xvf CIFAR-10-P.tar -C "."
  fi

  cd ..
fi

echo "Data preparation is done."


echo "================================================================"
echo "                  testing & analyzing the models                "
echo "================================================================"

python3 test.py $NOT_ANALYZE $OPT
