#include <stdio.h>
#include "floatfann.h"
#include <math.h>
double* evaluate(struct fann *ann, struct fann_train_data *data){
  fann_type *calc_out;
  fann_type input[2];
  int i;
  double setError0 = 0, setError1 = 0;
  static double ret[2];
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
      printf("Input:( %f, %f) -> Output:( %f, %f) -> Error:( %f, %f)\n",
	     input[0], input[1], calc_out[0], calc_out[1], 
	     fabs(data->output[i][0] - calc_out[0]), fabs(data->output[i][1] - calc_out[1]));

      setError0 += fabs(calc_out[0]); 
      setError1 += fabs(calc_out[1]);
    }
  //This could be calculated smarter
  printf("bit0 set error: %f \n", setError0);
  printf("bit1 set error: %f \n", setError1);
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
  printf("ann[0]:\n");
  evaluate(ann, data);

  printf("bnn[0]:\n");
  evaluate(bnn, data);

  printf("cnn[0]:\n");
  evaluate(cnn, data);

  
  fann_destroy(ann);
  fann_destroy(bnn);
  fann_destroy(cnn);
  
  return 0;
}
