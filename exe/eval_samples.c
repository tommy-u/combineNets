#include "fann.h"

//#define DEBUG                                                                                         
void evaluateSampleErrors(struct fann *ann, struct fann_train_data *data){
  //calc_out is just a pointer to the ann->output array                                                
  fann_type *calc_out, *inputData = *data->input, *outputData = *data->output;
  int i, j, numIn = fann_get_num_input(ann), numOut = fann_get_num_output(ann), lenTrain = fann_length_train_data(data);
  int correct_samples=0, incorrect_samples=0;
  unsigned char err_sample;
  //Loop over training samples.                          
  for(i = 0; i < lenTrain; i++, inputData += numIn){
    err_sample = 0;
    //Get predictions.                                                                                
    calc_out = fann_run(ann, inputData);
    //Loop over output nodes                                                                         
    for(j = 0; j < numOut; j++, calc_out++, outputData++ ) {
      //Assume nodewise error into array.                                                            
      if( (fabs(*outputData - *calc_out)) > .5)
	err_sample = 1;
    }
    err_sample ? incorrect_samples++ : correct_samples++;
  }
  printf("Incorrect %d Correct %d Percent Wrong %f \n", incorrect_samples, correct_samples, (double) incorrect_samples/(incorrect_samples+correct_samples));
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


int main(int argc, char *argv[])
{
  
  struct fann_train_data *data = NULL;
  struct fann **nets = NULL;
  int i, numNets = argc-2;

  data = checkArgs(argc, argv, data);
  nets = populateNets(numNets, argv, nets);

  for(i=0; i < numNets; i++){
    printf("%s \n", argv[i + 2]);
    evaluateSampleErrors(nets[i], data);
  }

  for (i = 0; i < numNets; i++)
    fann_destroy(nets[i]);
  free(nets);

  fann_destroy_train(data);
  
  return 0;

}


