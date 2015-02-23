The purpose of ctr is to combine two nets, ann & bnn into one net cnn. One of the output bits of cnn is predicted by the net strcuture prvided by ann and the other bit is provided by bnn. This iteration will automatically select the net with the min error for a given bit. Also, the input to hidden layer connections have been automated.

To compile:
make ctr cte

To run:
./ctr
./cte
