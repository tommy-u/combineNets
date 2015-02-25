iteration2. 
./ctr combines ann and bnn into cnn. Cnn has the property that each output bit is calculated fully by the network that predicts its value with the lowest error over the training set. After this is debugged, it should be possible to combine any 3 layer network with the same number of input and output nodes. The number of hidden notes should not matter. This is one step toward full generality. 

Issues:
Needs generalization in the evaluate function of both ctr and cte. Memory management is shooting from the hip and needs to be locked down. The work loop needs to be cleaned up so a human can read it. 

To compile:
make ctr cte

To run:
./ctr
./cte
