#include "fann.h"
#include <stdlib.h>
#include <assert.h>
#define DEBUGCONNECTIONS
//#define DEBUG
/*
  Combining n networks with same input and output size.  
  tommyu@bu.edu
*/
struct fann* init(unsigned int layers, unsigned int input, unsigned int hid, unsigned int hid2, unsigned int out, struct fann *ann){
  printf("%d %d %d %d %d \n",layers, input, hid, hid2, out);
  ann = fann_create_standard(layers, input, hid, hid2, out);
  fann_set_training_algorithm(ann, FANN_TRAIN_QUICKPROP);
  //  fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
  //For average.
  fann_set_activation_function_output(ann, FANN_LINEAR);
  fann_set_activation_steepness_output(ann, 1);
  return ann;
} 

struct fann ** populateNets(int numNets, char *argv[], struct fann **nets, int argc) {
if(argc < 2) {
    printf("Usage: ./committee_average <net1> ... <netn>  \n");
    printf("Supply one data file, then n nets for combining.\n");
    printf("Too few args. Exiting\n");
    exit(1);
  }
  nets = malloc ( (numNets )* sizeof(struct fann*));
  if(nets == NULL){
    printf("error allocating nets \n");
    exit(1);
  }

  int i;
  for (i = 0; i < numNets; i++) {
    nets[i] = fann_create_from_file(argv[i+1]);
    if(nets[i] == NULL){
      printf("error allocating nets[%d]",i+1);
      printf("mem leak 128 bytes");
      exit(1);
    }
  }
  return nets;
}
int associateNet(int conn, int numNets, int num_h1, unsigned int *hid_pos){
  int i;
  int in = conn % num_h1;
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
    }

  int *net_ctr;
  net_ctr = calloc ((numNets), sizeof(int));
  if (net_ctr ==NULL){ printf("malloc error \n"); exit(1); }

  int in_h1_con = (layers_cnn[0]+1) * layers_cnn[1];
  int h1_h2_con = (layers_cnn[1]+1) * layers_cnn[2];
  printf("in_h1_con: %d \n",in_h1_con);
  printf("h1_h2_con: %d \n",h1_h2_con);
  printf("layers_cnn[2]: %d\n",layers_cnn[2]);
  int total_connections = cnn->total_connections;
  int cnn_ctr = 0;

  //Input to hid1 connections
  for(i=0; i<numNets; i++){
    int in_h1_comp_con = (component_nets[i][0] + 1) * component_nets[i][1];
    for(j=0; j< in_h1_comp_con; j++)
      cnn->weights[cnn_ctr++] = nets[i]->weights[net_ctr[i]++];
  }
  assert(cnn_ctr == in_h1_con );

  //h1 to h2 conn
  for (i = in_h1_con; i < in_h1_con + h1_h2_con; i++){
    int net_selector = (i-in_h1_con) / ((layers_cnn[1]+1) * component_nets[0][2]);
    printf("cnn[%d] \n",i);
    if ( ((i-in_h1_con) % (layers_cnn[1]+1)) == layers_cnn[1]){
#ifdef DEBUGCONNECTIONS
      printf("bias \n");
#endif
      cnn->weights[i] = nets[net_selector]->weights[net_ctr[net_selector]++];


    }else if ( (associateNet(i-in_h1_con, numNets, layers_cnn[1] + 1, hid_pos) ) == net_selector )  {
#ifdef DEBUGCONNECTIONS
      printf("make it \n");
#endif
      cnn->weights[i] = nets[net_selector]->weights[net_ctr[net_selector]++];

    }else {
      cnn->weights[i] = 0;
    }
  } 

  int in_h2_con = in_h1_con + h1_h2_con;
  int num_output = layers_cnn[3];
  int num_hid2 = layers_cnn[2]+1;
  printf("num_hid2: %d \n", num_hid2);
  for(i=0; i<num_output; i++){
    for(j=0; j<num_hid2; j++){
      printf("j: %d \n",j);

      if (j % num_hid2 == num_hid2-1){
        printf("this is a bias connection \n");
        cnn->weights[(i*(layers_cnn[2]+1))+in_h2_con+j] = 0;
      }else if(j % num_output == i){
        printf("connect\n");
        cnn->weights[(i*(layers_cnn[2]+1))+in_h2_con+j] = 1/(float)numNets;
      }else{
        printf("not connected \n");
        cnn->weights[(i*(layers_cnn[2]+1))+in_h2_con+j] = 0;
      }

    }
  }

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
  int num_output = fann_get_num_output(nets[0]);
  int i;
  unsigned int *neuron_arr;
  int total_hidden = 0;
  //Figure out total num hidden neurons

  for (i=0; i<numNets; i++){
    neuron_arr = malloc( (num_layers * sizeof(unsigned int)));
    if (neuron_arr == NULL) { printf("malloc error \n"); exit (1); }
    fann_get_layer_array(nets[i], neuron_arr);
    total_hidden += neuron_arr[1];
    free(neuron_arr);
  }
  
  cnn = init(num_layers + 1, num_input, total_hidden, numNets * num_output, num_output, cnn);

  copyWeightsAverage(nets, numNets, cnn);

  return cnn;
}

int main(int argc, char *argv[]) {
  int i;
  int numNets = argc -1;
  struct fann *cnn = NULL;
  struct fann **nets = NULL;
  //printf("numNets: %d\n",numNets);
  nets = populateNets(numNets, argv, nets, argc);
  cnn = averageNets(nets, numNets, cnn);

#ifdef DEBUGCONNECTIONS
  
  for(i = 0; i < cnn->total_connections; i++) {
    printf("weight %d = %f \n", i, cnn->weights[i]);
  }
#endif
  for(i=0; i<numNets; i++)
    free(nets[i]);
  fann_save(cnn, "avg.net");
  //Save & clean
  free(nets);
  fann_destroy(cnn);
 
  return 0;   
}
