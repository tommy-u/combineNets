#include "fann.h"
#include <stdio.h>
#include <time.h>
#include <stdlib.h>


int main(int argc, char *argv[]){
  if (argc != 3){
    printf("%d \n", argc);
    printf("usage <dat> <num_bootstraps> \n exiting \n");
    exit(1);
  }
  int i, j, k;
  struct fann_train_data *data = fann_read_train_from_file(argv[1]);
  struct fann_train_data *tmp_net = fann_read_train_from_file(argv[1]);
  int num_bstraps = atoi(argv[2]);
  char train_name[256];
  srand(time(NULL));
  fann_type * s = *data->input;
  fann_type * t = *data->output; 
  fann_type * u = *tmp_net->input;
  fann_type * v = *tmp_net->output; 

  for(i=0; i<num_bstraps; i++){
    //Start with a copy, then modify input and output vals. 
    for(j=0; j<data->num_data; j++){
      //Select a row
      int r = rand() % data->num_data;
      //Input
      for(k=0; k<data->num_input; k++)
        u[data->num_input * j + k] = s[data->num_input * r + k];  
        // printf("data->input[%d] = %f \n",j, s[3*r + k]);

      for(k=0; k<data->num_output; k++)
        v[data->num_output * j + k] = t[data->num_output * r + k];  
        // printf("data->output[%d] = %f \n",j, t[3*r + k]);
    } 
//    fann_shuffle_train_data(tmp_net);
    snprintf(train_name, sizeof train_name, "train%d", i);
    fann_save_train(tmp_net, train_name);
  }
 //Why mem error when this commented in? 
  fann_destroy_train(tmp_net);        
  fann_destroy_train(data);
}








