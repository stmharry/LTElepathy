#!/bin/bash

SRSLTE_PATH=/home/stmharry/Library/srsLTE
UHD_PATH=/home/stmharry/Library/uhd
FREQ=915e6
IS_CUSTOM=true

cd $SRSLTE_PATH/build/signal-kinetics \
    && make main

$UHD_PATH/host/build/examples/tx_waveforms \
    --args type=usrp2 \
    --rate 1e6 \
    --freq $FREQ \
    --gain 0 \
    2>&1 > /dev/null &

if [ $IS_CUSTOM == true ]
then
    $SRSLTE_PATH/build/lib/examples/pdsch_enodeb \
        -a type=x300,addr=192.168.10.3 \
        -f $FREQ \
        2>&1 > /dev/null &
fi

$SRSLTE_PATH/build/signal-kinetics/main \
    -a type=x300,addr=192.168.10.2 \
    -f 0 \
    -h 127.0.0.1 \
    -p 6006

killall -s SIGKILL tx_waveforms pdsch_enodeb
