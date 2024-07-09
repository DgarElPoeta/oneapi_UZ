//
// Created by radon on 9/10/20.
//

#ifndef SCHEDULERS_H
#define SCHEDULERS_H

#define DEBUG(x) if (debug) { std::cout << x << std::endl; }
#define DEVICE_DEBUG(x) if (debug) { std::cout << device_type << ": " << x << std::endl; }
#define PRINT_TIME auto tBefore = std::chrono::high_resolution_clock::now();\
   auto diffBefore = (tBefore - tStart).count();\
   auto diffBeforeS = diffBefore / 1e9;\
   std::cout << std::this_thread::get_id() << "->"<< diffBeforeS << std::endl;


#endif //SCHEDULERS_H
