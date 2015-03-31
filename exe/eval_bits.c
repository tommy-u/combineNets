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

struct fann_train_data * checkArgs(int argc, char *argv[], struct fann_train_data *data){
  if(argc <= 2 ) {
    printf("Usage: ./eval  <data> <net1> ... <netn>\n");
    printf("Supply training data file and at least one net for per output node error evaluation.\n");
    printf("Too few args. Exiting.\n");
    exit(1);
  }
  data = fann_read_train_from_file(argv[1]);
  if(data== NULL) {
    printf("error opening data file \n");
    //Think this mem leak is fann.
    exit(1);
  }
  return data;
}
struct fann ** populateNets(int numNets, char *argv[], struct fann **nets) {
  int numOutput = -1;

  nets = malloc ( numNets * sizeof(struct fann*));
  if(nets == NULL){
    printf("error allocating nets \n");
    exit(1);
  }

  int i;
  for (i = 0; i < numNets; i++){
    nets[i] = fann_create_from_file(argv[i+2]);
    //Check nets have same num output.
    if(numOutput < 0)
      numOutput = fann_get_num_output(nets[i]);
    else
      if (numOutput != fann_get_num_output(nets[i])){
        printf("net %d does not have the same number of outputs as net %d \n",i,i-1);
      }
    if(nets[i] == NULL){
      printf("error allocating nets[%d]",i+2);
      exit(1);
    }
  }
  return nets;
}
void printResults(int numNets, char *argv[], struct fann **nets, struct fann_train_data *data, double *errorArr) {
  int i, j;
  double *totals;

  totals = calloc(numNets, sizeof(double*));
  if(totals == NULL){
    printf("calloc failed\n");
    exit(1);
  }
  
  unsigned int numOutput= fann_get_num_output(nets[0]); 
  printf("Net \t ");
  for(i = 0; i < numOutput; i++)
    printf("bit %d error\t",i);
  printf("total error\n");

  for (i = 0; i < numNets; i++){
    errorArr = evaluateBitErrors(nets[i], data, errorArr);    
    printf("%s \t",argv[i + 2]); 
    for(j = 0; j < numOutput; j++){
      printf("%f \t",errorArr[j]);
      totals[i] += errorArr[j];
    }
    printf("%f \n",totals[i]);
    free(errorArr);
  }
  free(totals);
}


int main (int argc, char *argv[]) {
  //Assume same # ouput bits.
  struct fann_train_data *data = NULL;
  struct fann **nets = NULL;  
  double *errorArr = NULL;
  int i, numNets = argc-2;

  data = checkArgs(argc, argv, data);
  nets = populateNets(numNets, argv, nets);

  printResults(numNets, argv, nets, data, errorArr);


  for (i = 0; i < numNets; i++)
    fann_destroy(nets[i]);
  free(nets);
  //  free(errorArr);
  fann_destroy_train(data);
  return 0;
}
