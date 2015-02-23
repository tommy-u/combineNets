The purpose of ctr is to combine two nets, ann & bnn into one net cnn. One of the output bits of cnn is predicted by the net strcuture prvided by ann and the other bit is provided by bnn. Connections are hard coded for clarity & speed. Assumptions about net architecture are exploited, but room exists for full generality. 

To compile:
gcc ctr.c -Wall -lfann  -o ctr
gcc cte.c -Wall -lfann  -o cte

To run:
./ctr
./cte
