#include "fann.h"
#include <stdlib.h>
  struct fann* init(const unsigned int layers, const unsigned int input, const unsigned int hid, const unsigned int out, struct fann *ann){
    ann = fann_create_standard(layers, input, hid, out);
    fann_set_training_algorithm(ann, FANN_TRAIN_QUICKPROP);
    fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
    fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);
    return ann;
  } 
int main(){
//	const unsigned int input = 2, hid = 2, out = 2, layers = 3;
	const unsigned int input = 2, out = 2, layers = 3;	
	const float desired_error = (const float) 0.01;
	const unsigned int max_epochs = 500000;
	const unsigned int epochs_between_reports = 500;
	struct fann_train_data *data = fann_read_train_from_file("x.data");
	struct fann *ann = NULL, *bnn = NULL, *cnn;
	
	ann = init(layers, input, 2, out, ann);
	bnn = init(layers, input, 2, out, bnn);
	cnn = init(layers, input, 2, out, bnn);

	fann_train_on_file(ann, "x.data", max_epochs, epochs_between_reports, desired_error);
	fann_train_on_file(bnn, "x.data", max_epochs, epochs_between_reports, desired_error);
	fann_train_on_file(cnn, "x.data", max_epochs, epochs_between_reports, desired_error);

	fann_save(ann, "a.net");
    fann_save(bnn, "b.net");
    fann_save(cnn, "c.net");

    fann_destroy(ann);
    fann_destroy(bnn);
    fann_destroy(cnn);
    fann_destroy_train(data);

	return 0;
}