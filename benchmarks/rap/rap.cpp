Rap* rap = reinterpret_cast<Rap*>(opts->p_problem);
for (size_t i = offset; i<size+offset; ++i){
  int tmp,j;
  // int myid = get_global_id(0);
  if(i <= rap->M){
    tmp = rap->func[0];
    for(j=0; j<=i; j++){
      tmp = std::max(rap->a[i-j] + rap->func[j], tmp);
      rap->b[i] = tmp;
    }
  }
}
