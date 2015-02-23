#include <stdio.h>
#include "floatfann.h"
#include <math.h>
//#define DEBUG
double* evaluate(struct fann *ann, struct fann_train_data *data){
  fann_type *calc_out;
  fann_type input[2];
  int i;
  double error0, error1,setError0 = 0, setError1 = 0;
  static double ret[2];
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

int main()
{
  struct fann *ann = fann_create_from_file("a_xor_float.net");
  struct fann *bnn = fann_create_from_file("b_xor_float.net");
  struct fann *cnn = fann_create_from_file("c_xor_float.net");
  struct fann_train_data *data = fann_read_train_from_file("xor.data");
  //Assume same num input & output for ann, bnn, cnn.

  unsigned int numOutput;
  double *e1, *e2, *e3;
  numOutput = fann_get_num_output(ann);
  e1 = malloc( (numOutput * sizeof(double)));
  if (e1 == NULL){
    printf("malloc error \n");
    exit (1);
  }
  e2 = malloc( (numOutput * sizeof(double)));
  if (e2 == NULL){
    printf("malloc error \n");
    exit (1);
  }
  e3 = malloc( (numOutput * sizeof(double)));
  if (e3 == NULL){
    printf("malloc error \n");
    exit (1);
  }

#ifdef DEBUG      
  printf("ann[0]:\n");
#endif
  e1 = evaluate(ann, data);
//  printf("e[1] = %f, e[0] = %f \n",e1[1],e1[0]);
#ifdef DEBUG      
  printf("bnn[0]:\n");
#endif
  e2 = evaluate(bnn, data);
//  printf("e[1] = %f, e[0] = %f \n",e1[1],e1[0]);
#ifdef DEBUG      
  printf("cnn[0]:\n");
#endif
  e3 = evaluate(cnn, data);
//  printf("e[1] = %f, e[0] = %f \n",e1[1],e1[0]);

  
  fann_destroy(ann);
  fann_destroy(bnn);
  fann_destroy(cnn);
  //When I attempt to free these, it says pointer is not allocated, but 
  //I think they are... TODO fix this
  //printf("here1\n");
  //free(e1);
  //printf("here2\n");
  //free(e2);
  //printf("here3\n");
  //free(e3);
  return 0;
}
