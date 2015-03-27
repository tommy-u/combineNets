#include "fann.h"
#include <stdlib.h>
#include <assert.h>
//#define DEBUGCONNECTIONS
//#define DEBUG
/*
  Combining n networks with same input and output size.  
  tommyu@bu.edu
*/

  double* evaluateBitErrors(struct fann *ann, struct fann_train_data *data, double *errorArr){
  //calc_out is just a pointer to the ann->output array
    fann_type *calc_out, *inputData = *data->input, *outputData = *data->output;
    int i, j, numIn = fann_get_num_input(ann), numOut = fann_get_num_output(ann), lenTrain = fann_length_train_data(data);
    errorArr = calloc( (numOut), sizeof(double) );
    if (errorArr == NULL){
      printf("malloc error \n");
      exit (1);
    }
  //Loop over training samples.
    for(i = 0; i < lenTrain; i++, inputData += numIn)
    {
    //Get predictions.
      calc_out = fann_run(ann, inputData);
    //Loop over output nodes
      for(j = 0; j < numOut; j++, calc_out++, outputData++ ) {
      //Assume nodewise error into array.
      #ifdef DEBUG
        printf("bit %d error += %f \n", j, fabs(*outputData - *calc_out));
      #endif
        errorArr[j] += (double) fabs(*outputData - *calc_out);
      }
    }
    return errorArr;
  }

  struct fann* init(const unsigned int layers, const unsigned int input, const unsigned int hid, const unsigned int out, struct fann *ann){
    //    if(ann != NULL)
    // free(ann);
    ann = fann_create_standard(layers, input, hid, out);
    fann_set_training_algorithm(ann, FANN_TRAIN_QUICKPROP);
    fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
    fann_set_activation_function_output(ann, FANN_SIGMOID);
    return ann;
  } 

  void printConnTable(struct fann *ann, struct fann *bnn) {
  //Assumes same architecture.
    int i = 0;
    for(i = 0; i < ann->total_connections; i++)
    {
      printf("connection %d\n", i);
     // printf("from neuron: %d to neuron: %d \n", a_con[i].from_neuron, a_con[i].to_neuron);
      printf("weights:\n");
      printf("ann: \t \t bnn:\n");
     // printf("%f \t %f \n", a_con[i].weight, b_con[i].weight);
      printf("\n");
    }
  }
  void joinNets(struct fann *ann, struct fann *bnn, struct fann *cnn, unsigned int *numNeur_a, unsigned int *numNeur_b, double *e_a, double *e_b){
    int i, ct, j = 0, k = 0;
  //Assumes fully connected  
  //Assumes input & output same size.
  //Assumes one bias neuron in each non output layer (can this be change?).
    int numHiddenNeur = numNeur_a[1] + numNeur_b[1];
    int inToHidConn = (numNeur_a[0] + 1) * numHiddenNeur;

    for(i = 0; i < inToHidConn; i++) {
        //L0 -> L1 conn
      //Puts connections into correspondance with hidden neurons
      if (( i / (numNeur_a[0] + 1) ) < numNeur_a[1]  ){
        cnn->weights[i] = ann->weights[j++];

      }else {
        cnn->weights[i] = bnn->weights[k++];
      }
    }
    //L1 -> L2 conn
    int outBit;
    //Loop over output bits.
    for(outBit = 0; outBit < fann_get_num_output(cnn); outBit++){
      int oldJ = j, oldK = k;
      //Loop over arriving connections
      for(ct = 0; ct < numHiddenNeur + 1; ct ++) {
        //Determine from neuron
        int modHidden = (i - inToHidConn) % (numHiddenNeur + 1);
        //Determine which net minimizes error criterion.
        //Redundant for clarity
        if(e_a[outBit] < e_b[outBit])
        {
          //Add from ann.
          if(modHidden < numNeur_a[1]){
            cnn->weights[i++] = ann->weights[j++ + ((1 + numNeur_a[1]) * outBit)]; 
          //Zero bnn connections.
          }else if(modHidden >= numNeur_a[1] && modHidden < numHiddenNeur){
            cnn->weights[i++] = 0;
          //Don't forget bias. 
          }else{
            cnn->weights[i++] = ann->weights[j++ + ((1 + numNeur_a[1]) * outBit)]; 
          }
        }else{
          //Zero ann connections.
          if(modHidden < numNeur_a[1]){
            cnn->weights[i++] = 0;
          //Add from bnn.
          }else if(modHidden >= numNeur_a[1] && modHidden < numHiddenNeur){
            cnn->weights[i++] = bnn->weights[k++ + ((1 + numNeur_b[1]) * outBit ) ];           
          //Don't forget bias.
          }else{
            cnn->weights[i++] = bnn->weights[k++ + ((1 + numNeur_b[1]) * outBit ) ];           
          }
        }
      }
      //Can we elimate this for the general case?
      j = oldJ; k = oldK;
    }
  }

  struct fann* combineNets(struct fann *ann, struct fann *bnn, struct fann_train_data *data) {
    unsigned int num_layers, *numNeur_a, *numNeur_b; 
    struct fann *cnn = NULL;
    //Assumes ann and bnn have identical structure. 
    //New net will be 2x as many in hidden.
    num_layers = fann_get_num_layers(ann);
    if(num_layers != fann_get_num_layers(bnn)){
      printf("layers unequal \n");
      exit(1);
    }
   //Neuron counts
    numNeur_a = malloc( (num_layers * sizeof(unsigned int)));
    if (numNeur_a == NULL){
      printf("malloc error \n");
      exit (1);
    }
    numNeur_b = malloc( (num_layers * sizeof(unsigned int)));
    if (numNeur_b == NULL){
      printf("malloc error \n");
      exit (1);
    }
    fann_get_layer_array(ann, numNeur_a);
    fann_get_layer_array(bnn, numNeur_b);

    if( ( numNeur_a[0] != numNeur_b[0] ) || ( numNeur_a[num_layers-1] != numNeur_b[num_layers-1] ) ) {
      printf("num input or output unequal \n");
      exit (1);
    }  
   
    cnn = init(num_layers, numNeur_a[0], numNeur_a[1] + numNeur_b[1], numNeur_a[num_layers-1], cnn);


    //Error arrays
    double *e_a = NULL, *e_b= NULL;
    e_a = evaluateBitErrors(ann, data, e_a);
    e_b = evaluateBitErrors(bnn, data, e_b);

    //Fully connect net.
    joinNets(ann, bnn, cnn, numNeur_a, numNeur_b, e_a, e_b);
    
   
    free(e_a);
    free(e_b);
    free(numNeur_a);
    free(numNeur_b);  
    return cnn;
  }
  struct fann_train_data * checkArgs(int argc, char *argv[], struct fann_train_data *data){
    if(argc < 3) {
      printf("Usage: ./combine <data> <net1> ... <netn>  \n");
      printf("Supply one data file, then n nets for combining.\n");
      printf("Too few args. Exiting\n");
      exit(1);
    }
    data = fann_read_train_from_file(argv[1]);
    if(data== NULL) {
      printf("error opening data file, 128 byte mem leak\n");
    //Think this mem leak is fann again.
      exit(1);
    }
    return data;
  }
  struct fann ** populateNets(int numNets, char *argv[], struct fann **nets) {
    
    nets = malloc ( (numNets )* sizeof(struct fann*));
    if(nets == NULL){
      printf("error allocating nets \n");
      exit(1);
    }

    int i;
    for (i = 0; i < numNets; i++) {
      nets[i] = fann_create_from_file(argv[i+2]);
      if(nets[i] == NULL){
        printf("error allocating nets[%d]",i+2);
        printf("mem leak 128 bytes");
        exit(1);
      }
    }
    return nets;
  }

  int main(int argc, char *argv[]) {
    int i, numNets = argc -2;
    struct fann *cnn = NULL, *oldCnn = NULL;
    struct fann **nets = NULL;
    struct fann_train_data *data = NULL;

    data = checkArgs(argc, argv, data);
    nets = populateNets(numNets, argv, nets);
    cnn = nets[0];
    //There is a mem leak here.
    for(i = 0; i < numNets - 1; i++) {
      oldCnn = cnn;
      cnn = combineNets(cnn, nets[i+1], data);
      fann_destroy(oldCnn);
      fann_destroy(nets[i+1]);
    }

#ifdef DEBUGCONNECTIONS
    for(i = 0; i < cnn->total_connections; i++) {
      printf("weight %d = %f \n", i, cnn->weights[i]);
    }
#endif
    fann_save(cnn, "combined.net");
  //Save & clean
    free(nets);
    fann_destroy(cnn);
    fann_destroy_train(data);
    return 0;   
  }

