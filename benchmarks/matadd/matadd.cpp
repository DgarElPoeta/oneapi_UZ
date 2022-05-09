Matadd* matadd = reinterpret_cast<Matadd*>(opts->p_problem);
int masize = matadd->size;
for(size_t i=offset; i<offset+size; i++){
    for(size_t j=0; j<masize; j++)
        matadd->c[i * masize + j] = matadd->a[i * masize + j] + matadd->b[i * masize + j];
}
