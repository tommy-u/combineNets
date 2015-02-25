#include <stdio.h>
#include "floatfann.h"
#include <math.h>
#define DEBUG

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
       errorArr[j] += (double) fabs(*outputData - *calc_out);
    }
  }
  return errorArr;
}


int main()
{
  struct fann *ann = fann_create_from_file("a_xor_float.net");
  struct fann *bnn = fann_create_from_file("b_xor_float.net");
  struct fann *cnn = fann_create_from_file("c_xor_float.net");
  struct fann_train_data *data = fann_read_train_from_file("xor2.data");
  unsigned int numOutput= fann_get_num_output(ann);
  double *errorArr = NULL;
  int i;

  //Ann
  errorArr = evaluateBitErrors(ann, data, errorArr);
#ifdef DEBUG      
  printf("ann[0]:\n");
  for(i = 0; i < numOutput; i++)
    printf("error for bit %d is %f \n",i,errorArr[i]);
#endif
  free(errorArr);
  //Bnn
  errorArr = evaluateBitErrors(bnn, data,errorArr);
#ifdef DEBUG      
  printf("ann[0]:\n");
  for(i = 0; i < numOutput; i++)
    printf("error for bit %d is %f \n",i,errorArr[i]);
#endif
  free(errorArr);
 //Cnn
 errorArr = evaluateBitErrors(cnn, data,errorArr);
#ifdef DEBUG      
  printf("ann[0]:\n");
  for(i = 0; i < numOutput; i++)
    printf("error for bit %d is %f \n",i,errorArr[i]);
#endif
  free(errorArr);




  fann_destroy(ann);
  fann_destroy(bnn);
  fann_destroy(cnn);
  fann_destroy_train(data);
  return 0;
}
