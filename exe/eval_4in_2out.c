#include "fann.h"

int main(int argc,char *argv[]) {

  struct fann_train_data *data = fann_read_train_from_file(argv[1]);
  struct fann *ann = fann_create_from_file(argv[2]);
  
  printf("\nPerformance on training set ann \n");
  int i;
  fann_type*inputData = *data->input, *outputData = *data->output;
  fann_type *calc_out;
  for(i = 0; i < fann_length_train_data(data); i++, inputData+=2, outputData+=2){
    calc_out = fann_run(ann, inputData);
    printf("out actual: %f\toutcalculated: %f \tdiff: %f  ", outputData[0], *calc_out, (outputData[0] - *calc_out) );
    printf("out actual: %f\toutcalculated: %f \tdiff: %f  \n", outputData[1], calc_out[1], (outputData[1] - calc_out[0]) );
  }


  return 0;
}
