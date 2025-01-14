#!/bin/bash

STAGE="test"
MAXTSTS=4


declare -a PLACES=("gijon" "barcelona" "madrid" "paris" "newyorkcity" "london")
#declare -a PLACES=("london")
declare -a SEEDS=(100 12 8778 0 99968547 772 8002 4658 9 34785)

GPU=1
i=0

for WHERE in "${PLACES[@]}" ;do
  echo "$WHERE"

  for SEED in "${SEEDS[@]}" ;do

    #nohup sleep 5 > /dev/null 2>&1 &

    #MANUAL GPU
    nohup /usr/bin/python3.6 -u  Recomendation.py  -s $STAGE  -c "$WHERE" -gpu $GPU -seed $SEED --log2file > /dev/null 2>&1 &

    #AUTO GPU
    #nohup /usr/bin/python3.6 -u  Recomendation.py  -s $STAGE  -c "$WHERE"  -seed $SEED --log2file > /dev/null 2>&1 &

    # Almacenar los PID en una lista hasta alcanzar el máximo de procesos
    pids[${i}]=$!

    #GPU=$(($(($GPU+1%2))%2))
    i+=1

    echo "   -[$!] $WHERE $SEED"

    # Si se alcanza el máximo de procesos simultaneos, esperar
    if [ "${#pids[@]}" -eq $MAXTSTS ];
    then

      # Esperar a que acaben los 9
      for pid in ${pids[*]}; do
          wait $pid
      done
      pids=()
      i=0
    fi

    #Esperar 5 segundos entre pruebas
    sleep 5

  done

done