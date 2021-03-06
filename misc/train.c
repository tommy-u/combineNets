#include "fann.h"
//Tommy Unger (tommyu@bu.edu)
void build_committee_average(struct fann *ann, struct fann *bnn, struct fann *cnn){
  
  //  cnn = fann_create_standard(num_layers, num_input, num_output);  
  
}
int main(){
  const unsigned int num_input = 2;
  const unsigned int hid = 2;
  const unsigned int num_output = 2;
  const unsigned int num_layers = 3;
  const float desired_error = (const float) 0.001;
  const unsigned int max_epochs = 25;
  const unsigned int epochs_between_reports = 0;
  struct fann_train_data *data = fann_read_train_from_file("copy.dat");
  struct fann *ann = fann_create_standard(num_layers, num_input, hid, num_output);
  struct fann *bnn = fann_create_standard(num_layers, num_input, hid, num_output);  
  //  struct fann *cnn = fann_create_standard(num_layers + 1, num_input,hid, 2*num_output, num_output);  

  int i;
  fann_type*inputData = *data->input, *outputData = *data->output;

  fann_train_on_data(ann, data, max_epochs, epochs_between_reports, desired_error);
  fann_train_on_data(bnn, data, max_epochs, epochs_between_reports, desired_error);
  fann_type *calc_out;  
  //Check Performance

  printf("\nPerformance on training set ann \n");
  for(i = 0; i < fann_length_train_data(data); i++, inputData+=2, outputData+=2){
    calc_out = fann_run(ann, inputData);
    printf("out actual: %f\toutcalculated: %f \tdiff: %f  ", outputData[0], calc_out[0], (outputData[0]- calc_out[0]) );
    printf("out actual: %f\toutcalculated: %f \tdiff: %f  \n", outputData[1], calc_out[1], (outputData[1]- calc_out[1]) );
  }
  inputData = *data->input; outputData = *data->output;
  printf("\nPerformance on training set bnn \n");
  for(i = 0; i < fann_length_train_data(data); i++, inputData+=2, outputData+=2){
    calc_out = fann_run(bnn, inputData);
    printf("out actual: %f\toutcalculated: %f \tdiff: %f  ", outputData[0], calc_out[0], (outputData[0]- calc_out[0]) );
    printf("out actual: %f\toutcalculated: %f \tdiff: %f  \n", outputData[1], calc_out[1], (outputData[1]- calc_out[1]) );
  }
  /*
  inputData = *data->input; outputData = *data->output;
  printf("\nPerformance on training set bnn \n");
  for(i = 0; i < fann_length_train_data(data); i++, inputData+=2){
    calc_out = fann_run(bnn, inputData);
    printf("out actual: %f\toutcalculated: %f \tdiff: %f  \n", outputData[i], *calc_out, (outputData[i]- *calc_out) );
  }
  */
  /*
  cnn->weights[0] = ann->weights[0]; 
  cnn->weights[1] = ann->weights[1]; 
  cnn->weights[2] = ann->weights[2]; 

  cnn->weights[3] = bnn->weights[0]; 
  cnn->weights[4] = bnn->weights[1]; 
  cnn->weights[5] = bnn->weights[2]; 
  fann_set_activation_function_output(cnn, FANN_LINEAR);
  fann_set_activation_steepness_output(cnn, 1);
  cnn->weights[6] = .5;
  cnn->weights[7] = .5;
  cnn->weights[8] = 0;
  */
  /*
    inputData = *data->input; outputData = *data->output;
  printf("\nPerformance on training set committee \n");
  for(i = 0; i < fann_length_train_data(data); i++, inputData+=2){
    calc_out = fann_run(cnn, inputData);
    printf("out actual: %f\toutcalculated: %f \tdiff: %f  \n", outputData[i], *calc_out, (outputData[i]- *calc_out) );
  } 
  */
  /*
  printf("\nConnection values ann: \n");
  for(i = 0; i < fann_get_total_connections(ann); i++)
    printf("connection %d has weight %f \n", i, ann->weights[i]);
  printf("\nConnection values bnn: \n");
  for(i = 0; i < fann_get_total_connections(bnn); i++)
    printf("connection %d has weight %f \n", i, bnn->weights[i]);
  */

  fann_save(ann, "a.net");
  fann_save(bnn, "b.net");
  //fann_save(cnn, "committee_average");
  

  fann_destroy(ann);
  fann_destroy(bnn);
  //  fann_destroy(cnn);
  fann_destroy_train(data);

  return 0;
}
