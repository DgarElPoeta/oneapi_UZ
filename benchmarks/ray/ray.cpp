/*Ray* ray = reinterpret_cast<Ray*>(opts->p_problem);

std::vector<ptype> b2(size, 0);
bool verification_passed = true;

bool debug = false;

int show_errors = 5;
for (size_t i = 0; i<size; ++i){
  const ptype kernel_value = b[i];
  const ptype host_value = b2[i];
  if (kernel_value != host_value) {
    fprintf(stderr, "VERIFICATION FAILED for element %ld: %d != %d (kernel != host value)\n", i, kernel_value, host_value);
    show_errors--;
    verification_passed = false;
    if (show_errors == 0){
      break;
    }
  }
}
return verification_passed;*/
