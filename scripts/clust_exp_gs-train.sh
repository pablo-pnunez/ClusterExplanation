#!/bin/bash

MAXTSTS=6
GPU=0
STAGE=0 # GRIDSEARCH o TRAIN
i=1

declare -a CITIES=( "madrid" )
declare -a MODELS=( "a" "h" "hl5" )
declare -a PCTGS=( .25 )
declare -a LRATES=( 1e-4 5e-4 1e-3 )
declare -a BATCHES=( 128 256 512 ) # NO MAS DE 1024 en LND

declare -a CITIES=( "madrid" )
declare -a MODELS=( "hl5" )
declare -a PCTGS=( .25 )
declare -a LRATES=( 5e-4 1e-3 )
declare -a BATCHES=( 1024 2048 ) # NO MAS DE 1024


for CITY in "${CITIES[@]}" ;do
  echo "$CITY"

  for MODEL in "${MODELS[@]}" ;do
    echo "-$MODEL"

    for PCTG in "${PCTGS[@]}" ;do
      echo "--$PCTG"

      for LRATE in "${LRATES[@]}" ;do
        echo "---$LRATE"

        for BATCH in "${BATCHES[@]}" ;do
        echo "----$BATCH"

          SAVEPATH="out/cluster_explanation/gridsearch/"$CITY
          mkdir -p $SAVEPATH

          #MANUAL GPU
          nohup /usr/bin/python3.6 -u  ClusterExplanation.py  -gpu $GPU -s $STAGE -c $CITY -m $MODEL -p $PCTG -bs $BATCH -lr $LRATE > $SAVEPATH"/model_"$MODEL"_"$PCTG"_["$LRATE"_"$BATCH"].txt" &

          GPU=$(($(($GPU+1%2))%2))

          # Almacenar los PID en una lista hasta alcanzar el máximo de procesos
          pids[${i}]=$!
          i+=1

          echo "   -[$!] $MODEL"

          # Si se alcanza el máximo de procesos simultaneos, esperar
          if [ "${#pids[@]}" -eq $MAXTSTS ];
          then

            # Esperar a que acaben los X
            for pid in ${pids[*]}; do
                wait $pid
            done
            pids=()
            i=0
          fi

          #Esperar X segundos entre pruebas para que le de tiempo a ocupar memoria en GPU
          sleep 120

        done

      done

    done

  done

done