#ifndef MATADD_H
#define MATADD_H

typedef float ptype;

struct Matadd {
  std::vector<ptype> a;
  std::vector<ptype> b;
  std::vector<ptype> c;
  uint64_t size;
};

#endif //MATADD_H

