#include <stdio.h>
#include "fann.h"
//#define DEBUG
#define OUTPUT

double* evaluateBitErrors(struct fann *ann, struct fann_train_data *data, double *errorArr) {
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

int main (int argc, char *argv[])
{
  if(argc <= 2 ) {
    printf("Usage: ./outNodeEval  <data> <net>\n");
    printf("Supply a net and data for per output node error Evaluation.\n");
    printf("Too few args. Exiting.\n");
    exit(1);
  }
  if(argc > 3) {
    printf("Usage: ./eval <net> <data> \n");
    printf("Supply one net and one data file for per output node error evaluation.\n");
    printf("Too many input args. Exiting\n");
    exit(1);
  }

  struct fann_train_data *data = fann_read_train_from_file(argv[1]);
  if(data== NULL) {
    printf("error opening data file, 128 byte mem leak\n");
    //Think this mem leak is fann again.
    exit(1);
  }

  struct fann *ann = NULL;
  ann = fann_create_from_file(argv[2]);
  if(ann == NULL) {
    //128 bytes of mem are lost here. Think it's fann.
    //FANN-2.2.0-Source/src/fann_io.c in fann_create_from_file,
    //in error case, file is not closed. 
    printf("error opening ann file, 128 byte mem leak\n");
    exit(1);
  }

  double *errorArr = NULL;

#if defined(DEBUG) || defined(OUTPUT)
  unsigned int numOutput= fann_get_num_output(ann);
  int i;
#endif

  //Ann
  errorArr = evaluateBitErrors(ann, data, errorArr);

#ifdef OUTPUT    
  for(i = 0; i < numOutput; i++)
    printf("error for bit %d is %f \n",i,errorArr[i]);
#endif

  free(errorArr);
  fann_destroy(ann);
  fann_destroy_train(data);
  return 0;
}
