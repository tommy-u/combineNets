#include "fann.h"
#include <stdlib.h>
#define DEBUGCONNECTIONS

/*
  Combining 2 networks with same architecture.
  This is a minimal prototype. Connections are hardcoded for cnn. ann predicts output bit 0,
  bnn predicts output bit 1. This will be changed to allow for vote etc.
    
  Interested in: 1 determining a better error criterion. 2 considering over fitting. 
  3 allowing multiple connections from bias node so an exponentially weighted scheme can
  be attempted. 4 generalization: i any number of nets (easy?) ii any hidden layer structure
  but same input and output iii any input output relation iv any number of layers 
  (shortcut connections needed?)  
  
  Author: Tommy Unger (tommyu@bu.edu.)
  
*/

struct fann* init(const unsigned int layers, const unsigned int input, const unsigned int hid, const unsigned int out, struct fann **ann){
  *ann = fann_create_standard(layers, input, hid, out);
  fann_set_training_algorithm(*ann, FANN_TRAIN_QUICKPROP);
  fann_set_activation_function_hidden(*ann, FANN_SIGMOID_SYMMETRIC);
  fann_set_activation_function_output(*ann, FANN_SIGMOID_SYMMETRIC);
  return *ann;
} 

struct fann_connection* allocate(struct fann *ann, struct fann_connection **con) {
  *con = malloc( (ann->total_connections * sizeof(struct fann_connection) + 4));
  if (con == NULL){
    printf("malloc error \n");
    exit (1);
  }
  return *con;
}

void printConnTable(struct fann *ann, struct fann *bnn, struct fann_connection *a_con, struct fann_connection *b_con) {
  //Assumes same architecture.
  int i = 0;
  fann_get_connection_array(ann,a_con);
  fann_get_connection_array(bnn,b_con);
  for(i = 0; i < ann->total_connections; i++)
    {
      printf("connection %d\n", i);
      printf("from neuron: %d to neuron: %d \n", a_con[i].from_neuron, a_con[i].to_neuron);
      printf("weights:\n");
      printf("ann: \t \t bnn:\n");
      printf("%f \t %f \n", a_con[i].weight, b_con[i].weight);
      printf("\n");
    }
}
void joinNets(struct fann *ann, struct fann *bnn, struct fann *cnn, struct fann_connection *a_con, struct fann_connection *b_con, struct fann_connection *c_con, unsigned int *numNeur_a, unsigned int *numNeur_b){
  int i, j = 0, k = 0;
  
  //Assumes fully connected  
  //Assumes input & output same size.
  //Assumes one bias neuron in each non output layer (can this be change?).
  int numHiddenNeur = numNeur_a[1] + numNeur_b[1];
  int inToHidConn = (numNeur_a[0] + 1) * numHiddenNeur;
  int hidToOutConn = (numHiddenNeur + 1) * numNeur_a[2];
  fann_get_connection_array(cnn,c_con);
  for(i = 0; i < cnn -> total_connections; i++) {
    printf("%d, %d, %f \n", c_con[i].from_neuron, c_con[i].to_neuron, c_con[i].weight);      
    //L0 -> L1 conn
    if(i < inToHidConn){
      //Connecting TO neurons corresponding to ann when true
      if ( c_con[i].to_neuron > numNeur_a[0] && c_con[i].to_neuron <= numNeur_a[0] + numNeur_a[1]  ){
        printf("i = %d, from ann[%d]\n",i, j++);
      }else {
        printf("i = %d, from bnn[%d]\n",i, k++);
      }
    //L1 -> L2 conn
    }else{
      if ( c_con[i].from_neuron > numNeur_a[0] && c_con[i].from_neuron <= numNeur_a[0] + numNeur_a[1]  ){
        printf("i = %d, from ann[%d]\n",i, j++);
      }else {
        printf("i = %d, from bnn[%d]\n",i, k++);
      }
    }
  }
}

struct fann* combineNets(struct fann *ann, struct fann *bnn, struct fann_connection *a_con, struct fann_connection *b_con) {
  unsigned int num_layers, *numNeur_a, *numNeur_b; 
  struct fann *cnn;
  struct fann_connection *c_con;
  //Assumes ann and bnn have identical structure. 
  //New net will be 2x as many in hidden.
  num_layers = fann_get_num_layers(ann);

  if(num_layers != fann_get_num_layers(bnn)){
    printf("layers unequal \n");
    exit(1);
  }

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

  //assumes 3 layers
  cnn = init(num_layers, numNeur_a[0], numNeur_a[1] + numNeur_b[1], numNeur_a[num_layers-1], &cnn);
  c_con = allocate(cnn, &c_con);

  //Fully connect net, then nuke inferior connections. 
  joinNets(ann, bnn, cnn, a_con, b_con, c_con, numNeur_a, numNeur_b);
  free(numNeur_a);
  free(numNeur_b);  
  free(c_con);
  return cnn;
}

int main() {
  const unsigned int input = 2, hid = 2, out = 2, layers = 3;
  const float desired_error = (const float) 0.01;
  const unsigned int max_epochs = 500000;
  const unsigned int epochs_between_reports = 10;
  struct fann *ann, *bnn, *cnn;
  struct fann_connection *a_con, *b_con;

  
//Init  
  ann = init(layers, input, hid, out, &ann);
  bnn = init(layers, input, hid, out, &bnn);

  //Train
  fann_train_on_file(ann, "xor.data", max_epochs, epochs_between_reports, desired_error);
  fann_train_on_file(bnn, "xor.data", max_epochs, epochs_between_reports, desired_error);
  
  //Each connection struct is 2 ints and a fann_type. Here the fann_type is 
  //double, but sizeof(struct fann_connection) == 12. Is this a bug or 
  //expected? Would be right if fann_type were int. 
  a_con = allocate(ann, &a_con);
  b_con = allocate(bnn, &b_con);

#ifdef DEBUGCONNECTIONS
  printConnTable(ann, bnn, a_con, b_con);
#endif

  //Combine
  cnn = combineNets(ann, bnn, a_con, b_con);

#ifdef DEBUGCONNECTIONS
  int i;
  for(i = 0; i < cnn->total_connections; i++)
    {
      printf("weight %d = %f \n", i, cnn->weights[i]);
    }
#endif

  //Save & clean
  fann_save(ann, "a_xor_float.net");
  fann_save(bnn, "b_xor_float.net");
  fann_save(cnn, "c_xor_float.net");
  fann_destroy(ann);
  fann_destroy(bnn);
  fann_destroy(cnn);
  free(a_con);
  free(b_con);
  return 0;   
}

