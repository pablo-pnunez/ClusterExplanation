#!/bin/bash

BATCH=1024
MAXTSTS=6
STAGE=100

MODEL="hl5"
PCTG=.25
LRATE=5e-4

i=0

declare -a CITIES=( "london" )

declare -a IMG_SLC_MTHS=( 0 ) # Seleccionar las N_IMGS representativas mediante clustering (0) o aleatoriamente (1)
declare -a CS_MODES=( "all" )
declare -a ONLYPOS=( 0 ) # Todas las reviews (0) o solo positivas (1)

for CITY in "${CITIES[@]}" ;do
  echo "$CITY"

  for ONP in "${ONLYPOS[@]}" ;do
    echo "-$ONP"

    for IMG_SLC_MTH in "${IMG_SLC_MTHS[@]}" ;do
      echo "--$IMG_SLC_MTH"

      for CS_MODE in "${CS_MODES[@]}" ;do
        echo "---$CS_MODE"

        OUTPATH="out/cluster_explanation/test/"$CITY"/"$ONP
        mkdir -p $OUTPATH

        nohup /usr/bin/python3.6 -u  ClusterExplanation.py -s $STAGE -c $CITY -m $MODEL -p $PCTG -bs $BATCH -lr $LRATE -csm $CS_MODE -ism $IMG_SLC_MTH -onlypos $ONP > $OUTPATH"/"$CS_MODE"-["$IMG_SLC_MTH"].txt" &

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
        sleep 1 # 600


      done

    done

  done

done