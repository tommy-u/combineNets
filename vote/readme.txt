combine.c allows for any number of nets with the same number of output nodes to be combined. These nets can have arbitrary numbers of hidden nodes. 
Note, this is written to accommodate the ASC system and thus the output layer is sigmoid, not sigmoid symmetric. Outputs should be normalized to 0 to 1. 

eval.c allows an arbitrary number of nets to be evaluated based on their per output node error. This error term is simply the absolute value of the difference between their predicted output node value and the actual value from the training set evaluated over the entire training set.

errorArr [#output nodes];
loop over output nodes
	errorArr[this node] += abs(predicted val for node (run input through net) - actual valfor node (from training set ) )

To compile:
make all

To run:

Build Nets
./nets

Evaluate nets
./eval x.data a.net b.net c.net

Combine component nets.
./combine x.data a.net b.net c.net 

Check out combined net.
./eval x.data combined.net

