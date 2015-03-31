#include "fann.h"

int main(){
  struct fann *ann, *bnn;
  ann = fann_create_standard(3, 2, 2, 2);
  bnn = fann_create_standard(3, 2, 3, 2);
  fann_save(ann,"a.net");
  fann_save(bnn,"b.net");
  return 0;
}
