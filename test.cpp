/*
Test
*/
#include "lnet_test.h"
#include "lnet_logistic_regression_test.h"

using namespace lnet_test;
using namespace lnet_logistic_regression_test;

int main() {
  // test_random_gen();
  // bench();

  cout << "Test lnet regression\n";
  //test_regression();
  //test_regression_prostate();

  cout << "Test lnet logistic regression\n";
  test_logistic_regression();
}