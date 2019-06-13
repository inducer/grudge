#!/bin/bash

# Weak scaling: We run our code on one computer, then we buy a second computer
# and we can run twice as much code in the same amount of time.

# Strong scaling: We run our code on one computer, then we buy a second computer
# and we can run the same code in half the time.

# Examples:
# ./run_benchmark.sh -t WEAK -n 100 -r 20 -s 1000 -l ~/weak_scaling.dat -o weak_scaling.txt
# ./run_benchmark.sh -t STRONG -n 100 -r 20 -s 1000 -l ~/strong_scaling.dat -o strong_scaling.txt

set -eu

# NOTE: benchmark_mpi.py hangs when logfile is in a shared directory.
USAGE="Usage: $0 -t <WEAK|STRONG> -n num_elems -r order -s num_steps -l logfile -o outfile"
while getopts "t:n:r:s:l:o:" OPT; do
  case $OPT in
    t)
      case $OPTARG in
        WEAK)
          SCALING_TYPE='WEAK'
          ;;
        STRONG)
          SCALING_TYPE='STRONG'
          ;;
        *)
          echo $USAGE
          exit 1
          ;;
      esac
      ;;
    n)
      NUM_ELEMS=$OPTARG
      ;;
    r)
      ORDER=$OPTARG
      ;;
    s)
      NUM_STEPS=$OPTARG
      ;;
    l)
      LOGFILE=$OPTARG
      ;;
    o)
      OUTFILE=$OPTARG
      ;;
    *)
      echo $USAGE
      exit 1
      ;;
  esac
done


# NOTE: We want to make sure we run grudge in the right environment.
SHARED="/home/eshoag2/shared"
source $SHARED/miniconda3/bin/activate inteq
PYTHON=$(which python)
BENCHMARK_MPI="$SHARED/grudge/examples/benchmark_grudge/benchmark_mpi.py"

# Assume HOSTS_LIST is sorted in increasing order starting with one host.
HOSTS_LIST="\
porter \
porter,stout \
porter,stout,koelsch"

ENVIRONMENT_VARS="\
-x RUN_WITHIN_MPI=1 \
-x PYOPENCL_CTX=0 \
-x POCL_AFFINITY=1"

PERF_EVENTS="\
cpu-cycles,\
instructions,\
task-clock"

TEMPDIR=$(mktemp -d)
trap 'rm -rf $TEMPDIR' EXIT HUP INT QUIT TERM

echo "$(date): Testing $SCALING_TYPE scaling" | tee -a $OUTFILE

NUM_HOSTS=1
BASE_NUM_ELEMS=$NUM_ELEMS
for HOSTS in $HOSTS_LIST; do

  if [ $SCALING_TYPE = 'WEAK' ]; then
    NUM_ELEMS=$(echo $BASE_NUM_ELEMS $NUM_HOSTS | awk '{ print $1 * $2 }')
  fi

  BENCHMARK_CMD="$PYTHON $BENCHMARK_MPI $NUM_ELEMS $ORDER $NUM_STEPS $LOGFILE.trial$NUM_HOSTS"
  # NOTE: mpiexec recently updated so some things might act weird.
  MPI_CMD="mpiexec --output-filename $TEMPDIR -H $HOSTS $ENVIRONMENT_VARS $BENCHMARK_CMD"
  echo "Executing: $MPI_CMD"

  # NOTE: perf does not follow mpi accross different nodes.
  # Instead, perf will follow all processes on the porter node.
  echo "====================Using $NUM_HOSTS host(s)===================" >> $OUTFILE
  START_TIME=$(date +%s)
  perf stat --append -o $OUTFILE -e $PERF_EVENTS $MPI_CMD
  DURATION=$(($(date +%s) - $START_TIME))
  echo "Finished in $DURATION seconds"

  echo "===================Output of Python===================" >> $OUTFILE
  find $TEMPDIR -type f -exec cat {} \; >> $OUTFILE
  echo "======================================================" >> $OUTFILE
  rm -rf $TEMPDIR/*

  if [ $NUM_HOSTS -eq 1 ]; then
    BASE_DURATION=$DURATION
  fi

  # Efficiency is expected / actual
  if [ $SCALING_TYPE = 'STRONG' ]; then
    EFFICIENCY=$(echo $DURATION $BASE_DURATION $NUM_HOSTS | awk '{ print $2 / ($3 * $1) * 100"%" }')
  elif [ $SCALING_TYPE = 'WEAK' ]; then
    EFFICIENCY=$(echo $DURATION $BASE_DURATION | awk '{ print $2 / $1 * 100"%" }')
  fi

  echo "Efficiency for $SCALING_TYPE scaling is $EFFICIENCY for $NUM_HOSTS host(s)." | tee -a $OUTFILE

  ((NUM_HOSTS++))
done
