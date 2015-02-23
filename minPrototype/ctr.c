#include "fann.h"
#include <stdlib.h>
//#define DEBUGCONNECTIONS

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

struct fann* combineNets(const unsigned int input, const unsigned int hid, const unsigned int out, const unsigned int layers, struct fann_connection *a_con, struct fann_connection *b_con) {
  int i,j=0,k=0;
  struct fann *cnn;
  //Assumes ann and bnn have identical structure. 
  //New net will be 2x as many in hidden.
  cnn = init(layers, input, hid*2, out, &cnn);

  //hard coding my solution, sorry for what you're about to witness
  //j walks a_conn[j].weight, k walks b_conn[k].weight

  //Writing this out pedantically so I don't make a dumb mistake.
  for(i = 0; i < cnn -> total_connections; i++)
    {
      if(i <= 5) {
	cnn->weights[i] = a_con[j++].weight;
	continue;
      }
      else if(i >= 6 && i <= 11){
	cnn->weights[i] = b_con[k++].weight;
	continue;
      }
      else if(i == 12 || i == 13)
	cnn->weights[i] = a_con[j++].weight;
      //also need to advance k so we skip those connections
      else if(i == 14 || i == 15){
	cnn->weights[i] = 0;
	k++;
      }
      else if(i == 16)
	{
	cnn->weights[i] = a_con[j++].weight;
	k++;
	}
      else if(i == 17 || i == 18)
	cnn->weights[i] = 0;     
      else if(i >= 19 && i <= 21)
	cnn->weights[i] = b_con[k++].weight;  
    }
  return cnn;
}

int main() {
  const unsigned int input = 2, hid = 2, out = 2, layers = 3;
  const float desired_error = (const float) 0.01;
  const unsigned int max_epochs = 500000;
  const unsigned int epochs_between_reports = 10;
  struct fann *ann, *bnn, *cnn;
  struct fann_connection *a_con, *b_con;
  int i;
  
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
  cnn = combineNets(input, hid, out, layers, a_con, b_con);

#ifdef DEBUGCONNECTIONS
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

