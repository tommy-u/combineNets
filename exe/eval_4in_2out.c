#include "fann.h"
void print_per_sample_per_node_error(struct fann *ann, struct fann_train_data *data, char *argv[]){
  printf("\nPerformance of %s on training set %s \n",argv[2], argv[1]);
  int i, j;
  fann_type*inputData = *data->input, *outputData = *data->output;
  fann_type *calc_out;
  int num_output = fann_get_num_output(ann); 
  for(i = 0; i < fann_length_train_data(data); i++, inputData+=num_output, outputData +=num_output){
    printf("sample: %d\n",i);
    calc_out = fann_run(ann, inputData);
  for(j=0; j<num_output; j++){
    printf("output %d: ",j);
    printf("out actual: %f\toutcalculated: %f \tdiff: %f  \n", outputData[j], calc_out[j], (outputData[j] - calc_out[j]) );
}
  printf("\n");
  }
}
int main(int argc,char *argv[]) {

  struct fann_train_data *data = fann_read_train_from_file(argv[1]);
  struct fann *ann = fann_create_from_file(argv[2]);
  
  print_per_sample_per_node_error(ann,data, argv);  


  return 0;
}
