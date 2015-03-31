#include "fann.h"
#include <stdlib.h>
#include <assert.h>
#define DEBUGCONNECTIONS
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

  struct fann* init(unsigned int layers, unsigned int input, unsigned int hid, unsigned int hid2, unsigned int out, struct fann *ann){
    //printf("fpe\n");
    printf("%d %d %d %d %d \n", layers, input, hid, hid2, out);
    ann = fann_create_standard(layers, input, hid, hid2, out);
    fann_set_training_algorithm(ann, FANN_TRAIN_QUICKPROP);
    fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
    //How we pull off the average
    fann_set_activation_function_output(ann, FANN_LINEAR);
    fann_set_activation_steepness_output(ann, 1);
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

//    cnn = init(num_layers, numNeur_a[0], numNeur_a[1] + numNeur_b[1], numNeur_a[num_layers-1], cnn);


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
int associateNet(int conn, int numNets, int num_h1, unsigned int *hid_pos){
  int i;
  int in = conn % num_h1;
 // printf("%d, %d, %d \n",conn, numNets, num_h1);
  //Should always break
  for(i=0; i<numNets; i++)
    if (hid_pos[i] > in)
      break;

  return i;
  }
  void copyWeightsAverage(struct fann **nets, int numNets, struct fann *cnn){

    int i,j;
    unsigned int *layers_cnn;
    //For the component nets. 
    int num_layers = 3;
    //loop over input nets 
    layers_cnn = malloc( (num_layers + 1) * sizeof(unsigned int));
    if (layers_cnn == NULL) { printf("malloc error \n"); exit (1); }
    
    fann_get_layer_array(cnn, layers_cnn);

    //Cnn pos of last neuron in L1 (hidden layer of comp nets)
    unsigned int *hid_pos = malloc ( numNets * sizeof(unsigned int));
    if (hid_pos == NULL) { printf("malloc error \n"); exit (1); }

    unsigned int **component_nets;
    component_nets = malloc( numNets * sizeof(unsigned int*));
    if (component_nets == NULL) { printf("malloc error \n"); exit (1); }
    for(i = 0; i<numNets; i++)
    {
      component_nets[i] = malloc ( num_layers * sizeof(unsigned int));
      if (component_nets[i] == NULL) { printf("malloc error \n"); exit (1); }
      //Used in setting hid1 to hid2 conn
      fann_get_layer_array(nets[i], component_nets[i]); 
      hid_pos[i] = ((i==0) ? component_nets[i][1] : hid_pos[i-1] + component_nets[i][1]);
      //printf("alkjhsdf\n");
      //printf("%d \n",hid_pos[i] );
    }

    int *net_ctr;
    net_ctr = calloc ((numNets), sizeof(int));
    if (net_ctr ==NULL){ printf("malloc error \n"); exit(1); }

    int in_h1_con = (layers_cnn[0]+1) * layers_cnn[1];
    int h1_h2_con = (layers_cnn[1]+1) * layers_cnn[2];
    int total_connections = cnn->total_connections;
    int cnn_ctr = 0;

  //Build input to hid1 layer connections
    for(i=0; i<numNets; i++){
      int in_h1_comp_con = (component_nets[i][0] + 1) * component_nets[i][1];
      for(j=0; j< in_h1_comp_con; j++)
        cnn->weights[cnn_ctr++] = nets[i]->weights[net_ctr[i]++];
    }
    assert(cnn_ctr == in_h1_con );
  //h1 to h2 conn
  //Zero connections
    // for (i=in_h1_con; i< in_h1_con + h1_h2_con; i++)
    //   cnn->weights[i] = 0;

    for (i = in_h1_con; i < in_h1_con + h1_h2_con; i++){
      int net_selector = (i-in_h1_con) / ((layers_cnn[1]+1) * component_nets[0][2]);
//    printf("i: %d, i-in_h1_con: %d, layers_cnn[1]: %d\n",i, i-in_h1_con ,layers_cnn[1] );
    //Bias case. Adding one because fann ... layers_array doesn't include it.

      if ( ((i-in_h1_con) % (layers_cnn[1]+1)) == layers_cnn[1]){
        printf("cnn[%d] \n",i);
        printf("bias \n");
        cnn->weights[i] = nets[net_selector]->weights[net_ctr[net_selector]++];

//        cnn->weights[i] = 
      }else if ( (associateNet(i-in_h1_con, numNets, layers_cnn[1] + 1, hid_pos) ) == net_selector )  {
        printf("cnn[%d] \n",i);
        printf("make it \n");
        cnn->weights[i] = nets[net_selector]->weights[net_ctr[net_selector]++];

//        if()
        //printf("%d \n", associateNet(i-in_h1_con, numNets, layers_cnn[1] + 1, hid_pos) );
        
//printf("%d \n", (i-in_h1_con) / ((layers_cnn[1]+1) * component_nets[0][2]));
      
      }else {
        cnn->weights[i] = 0;
      }
    } 
int in_h2_con = in_h1_con + h1_h2_con;
for(i=in_h1_con + h1_h2_con; i< total_connections; i++ )
    if ( ((i-in_h2_con) % (layers_cnn[2]+1)) == layers_cnn[2]){
        printf("cnn[%d] \n",i);
        printf("bias \n");
        cnn->weights[i] = 0;        
    }else{
      printf("cnn[%d] \n",i);
      printf("make it\n");

      cnn->weights[i] = 1/((double)numNets);
}



  //Build hid1 to hid2 layer connections
  //Assume all input nets have same number outputs
  //  unsigned int num_output_component = component_nets[0][2];
 //   unsigned int total_hid2_neur = num_output_component * numNets;
/*
  for(i=((layers_cnn[0] + 1) * layers_cnn[1]) - 1 ; i< (layers_cnn[1] + 1) * layers_cnn[2] ; i++)
    cnn->weights[cnn_ctr++] = 0;

  for(i=0; i<total_hid2_neur; i++){

    continue;
  }
*/

  
  //for (i=0; i< in_h1_con; i++)
    //cnn->weights[i] = 0;
  
  

  

  for(i = 0; i<numNets; i++)
    free(component_nets[i]); 
  free(hid_pos);
  free(layers_cnn);
  free (component_nets);
  free(net_ctr);
}



  struct fann* averageNets(struct fann **nets, int numNets, struct fann *cnn ){
    //Assume inputs are 3 layers, new net will be 4

    int num_layers = 3;
    //Assume nets have same number of input and output nodes.
    int num_input = fann_get_num_input(nets[0]);
    int num_output = fann_get_num_input(nets[0]);
    int i;
    unsigned int *neuron_arr;
    int total_hidden = 0;
    //Figure out total num hidden neurons

    for (i=0; i<numNets; i++){
      neuron_arr = malloc( (num_layers * sizeof(unsigned int)));
      if (neuron_arr == NULL) { printf("malloc error \n"); exit (1); }
      fann_get_layer_array(nets[i], neuron_arr);
    //  printf("%d, %d, %d \n", neuron_arr[0],neuron_arr[1],neuron_arr[2]  );
      total_hidden += neuron_arr[1];
      free(neuron_arr);
    }
      //printf("%d \n", fann_get_num_input( nets[0]));    
//  printf("total_hidden: %d \n",total_hidden);
    cnn = init(num_layers + 1, num_input, total_hidden, numNets * num_output, num_output, cnn);

    copyWeightsAverage(nets, numNets, cnn);

    return cnn;
  }

  int main(int argc, char *argv[]) {
  //printf("main\n");
    int numNets = argc -2;
    struct fann *cnn = NULL;
    struct fann **nets = NULL;
    struct fann_train_data *data = NULL;

    data = checkArgs(argc, argv, data);
    nets = populateNets(numNets, argv, nets);



  //printf("avg now\n");
    cnn = averageNets(nets, numNets, cnn);
//  printf("done avg \n");

#ifdef DEBUGCONNECTIONS
    int i;
    for(i = 0; i < cnn->total_connections; i++) {
      printf("weight %d = %f \n", i, cnn->weights[i]);
    }
#endif


    fann_save(cnn, "avg.net");
  //Save & clean
    free(nets);
    fann_destroy(cnn);
    fann_destroy_train(data);
    return 0;   
  }

//loop over receiving nodes. Don't include bias
  // for (i=0; i< layers_cnn[2];  i++){
  //   // printf("&& %d \n", i);
  //   // int net_selector = i / layers_cnn[3];
  //   // int num_hidden_comp = component_nets[net_selector][1];

  //   for (j=0; j< num_hidden_comp + 1; j++){
  //       //Normal connection, or bias?
  //     printf("** %d \n",j);
  //     if(j != (num_hidden_comp) ){

        //printf("cnn[%d] = comp[%d][%d]\n",in_h1_con + (i * (layers_cnn[1] + 1) * layers_cnn[3]) + j, net_selector,net_ctr[net_selector]  );
        //printf("cnn[%d] = comp[%d][%d]\n",in_h1_con + (i * (layers_cnn[1] + 1) * layers_cnn[3]) + j, net_selector,net_ctr[net_selector]  );

        //cnn->weights[in_h1_con + (net_selector * (layers_cnn[1] + 1) * layers_cnn[3]) + j] = nets[net_selector]->weights[net_ctr[net_selector]++];
      //}else{
        //printf("bias \n");
        //printf("cnn[%d] = comp[%d][%d] \n", (i * layers_cnn[1]) + layers_cnn[1] + in_h1_con, i,net_ctr[i]);
        //printf("cnn[%d] = comp[%d][%d] \n", (i * layers_cnn[1]) + layers_cnn[1] + in_h1_con, i,net_ctr[i]);
        //cnn->weights[(i * layers_cnn[1]) + layers_cnn[1] + in_h1_con]  = nets[i]->weights[net_ctr[i]++];

  
      //if 


