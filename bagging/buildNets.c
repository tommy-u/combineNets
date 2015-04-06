#include "fann.h"
#include <stdlib.h>
//#define DEBUG
void evaluateBitErrors(struct fann *ann, struct fann_train_data *data){
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
#ifdef DEBUG
        printf("bit %d error += %f \n", j, fabs(*outputData - *calc_out));
#endif
	if( (fabs(*outputData - *calc_out)) > .5)
	  err_sample = 1;
      }
      err_sample ? incorrect_samples++ : correct_samples++;
      
  }
  printf("Incorrect %d Correct %d Percent Wrong %f \n", incorrect_samples, correct_samples, (double) incorrect_samples / (incorrect_samples+correct_samples)); 
}

int main(int argc, char *argv[]){
  //Build one net for input training file. 
  if(argc != 3){
    printf("usage ./buildNet <data> <num_nets>");
    exit(1);
  }
  int i;

  struct fann_train_data *data = fann_read_train_from_file(argv[1]);
  int num_nets = atoi(argv[2]);
  //Danger
  char train_name[256];
  const unsigned int num_input = data->num_input;
  const unsigned int num_output = data->num_output;
  //Arbitrary
  const unsigned int num_layers = 3;
  const unsigned int num_neurons_hidden = 3;
  const float desired_error = (const float) 0.001;
  const unsigned int max_epochs = 3;
  const unsigned int epochs_between_reports = 0;
  for(i=0; i<num_nets; i++){
  struct fann *ann = fann_create_standard(num_layers, num_input, num_neurons_hidden, num_output);
  fann_set_training_algorithm(ann, FANN_TRAIN_QUICKPROP);
  fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
  fann_set_activation_function_output(ann, FANN_SIGMOID);
  fann_train_on_file(ann, argv[1], max_epochs, epochs_between_reports, desired_error);
  snprintf(train_name, sizeof train_name, "net%d(%s)",i, argv[1]);
  evaluateBitErrors(ann, data);
  fann_save(ann, train_name);
  fann_destroy(ann);
}
 fann_destroy_train(data);
  return 0;

}













