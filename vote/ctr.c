#include "fann.h"
#include <stdlib.h>
//#define DEBUGCONNECTIONS
//#define DEBUG
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
double* evaluate(struct fann *ann, struct fann_train_data *data){
  //TODO make this code general
  fann_type *calc_out;
  fann_type input[2];
  int i;
  double error0, error1,setError0 = 0, setError1 = 0, *ret;
  ret = malloc( sizeof(double) * (fann_get_num_output(ann)));
  if (ret == NULL){
    printf("malloc error \n");
    exit (1);
  }
  
#ifdef DEBUG
  printf("bits listed high bit -> low bit \n");
#endif
  for(i = 0; i< 4; i++)
    {
      if(i%2 == 0)
  input[0] = -1;
      else
  input[0] = 1;
      
      if(i/2 == 0)
  input[1] = -1;
      else
  input[1] = 1;
      calc_out = fann_run(ann, input);
      error0 = fabs(data->output[i][0] - calc_out[0]);
      error1 = fabs(data->output[i][1] - calc_out[1]);
#ifdef DEBUG      
      printf("Input:( %f, %f) -> Output:( %f, %f) -> Error:( %f, %f)\n",
       input[1], input[0], calc_out[1], calc_out[0], 
       error1, error0);
#endif
      setError0 += error0; 
      setError1 += error1;
    }
  //This could be calculated smarter
#ifdef DEBUG      
  printf("bit0 set error: %f \n", setError0);
  printf("bit1 set error: %f \n", setError1);
#endif
  ret[0] = setError0;
  ret[1] = setError1;
  return ret;
}
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
void joinNets(struct fann *ann, struct fann *bnn, struct fann *cnn, struct fann_connection *a_con, struct fann_connection *b_con, struct fann_connection *c_con, unsigned int *numNeur_a, unsigned int *numNeur_b, double *e_a, double *e_b){
  int i, ct, j = 0, k = 0;
  int out = fann_get_num_output(ann);
  //Assumes fully connected  
  //Assumes input & output same size.
  //Assumes one bias neuron in each non output layer (can this be change?).
  int numHiddenNeur = numNeur_a[1] + numNeur_b[1];
  int inToHidConn = (numNeur_a[0] + 1) * numHiddenNeur;
//  int hidToOutConn = (numHiddenNeur + 1) * numNeur_a[2];
  fann_get_connection_array(cnn,c_con);
  for(i = 0; i < inToHidConn; i++) {
    printf("%d, %d, %f \n", c_con[i].from_neuron, c_con[i].to_neuron, c_con[i].weight);      
    //L0 -> L1 conn
    if(i < inToHidConn){
      //Connecting TO neurons corresponding to ann when true
      if ( c_con[i].to_neuron > numNeur_a[0] && c_con[i].to_neuron <= numNeur_a[0] + numNeur_a[1]  ){
        printf("i = %d, from ann[%d]\n",i, j++);
      }else {
        printf("i = %d, from bnn[%d]\n",i, k++);
      }
    }
  }
  //L1 -> L2 conn
  //Loop over output bits
  printf("hidden -> output\n");
  for(i = 0; i < numNeur_a[out -1]; i++) {
    //Pick net with lower error for that bit
    printf("e_a[i] = %f, e_b[i] = %f \n", e_a[i], e_b[i]);
    if(e_a[i] < e_b[i]){
      printf("true \n");
      for(ct = 0; ct < out; ct ++){
        //Copy weight from ann
        printf("i = %d, from ann[%d]\n",i, j++);
                //Zero contribution from bnn
        //Advance weight counter for both component nets
      }
      printf("grab bias from ann");
    }else {
      printf("false\n");
      for(ct = 0; ct < out; ct ++){
        //Copy weight from bnn
        printf("i = %d, from bnn[%d]\n",i, j++);
      }
      printf("grab bias from bnn \n");
    }
  }
}
struct fann* combineNets(struct fann *ann, struct fann *bnn, struct fann_connection *a_con, struct fann_connection *b_con, struct fann_train_data *data) {
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
  //assumes 3 layers
  cnn = init(num_layers, numNeur_a[0], numNeur_a[1] + numNeur_b[1], numNeur_a[num_layers-1], &cnn);
  c_con = allocate(cnn, &c_con);

  //Error arrays
  double *e_a, *e_b;
  e_a = malloc( (numNeur_a[num_layers-1] * sizeof(double)));
  if (e_a == NULL){
    printf("malloc error \n");
    exit (1);
  }
  e_b= malloc( (numNeur_a[num_layers-1] * sizeof(double)));
  if (e_b == NULL){
    printf("malloc error \n");
    exit (1);
  }
  printf("here 123\n");
  e_a = evaluate(ann, data);
  e_b = evaluate(bnn, data);
  printf("%p, %p \n", ann, bnn);
  printf("*** %f, %f \n ", e_a[0], e_b[0]);  

  //Fully connect net.
  joinNets(ann, bnn, cnn, a_con, b_con, c_con, numNeur_a, numNeur_b, e_a, e_b);
  free(numNeur_a);
  free(numNeur_b);  
  //IDK about this. TODO ask someone smarter. 
  // free(e_a);
  // free(e_b);
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
  struct fann_train_data *data = fann_read_train_from_file("xor.data");
  
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
  cnn = combineNets(ann, bnn, a_con, b_con, data);

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
  fann_destroy_train(data);
  free(a_con);
  free(b_con);
  return 0;   
}

