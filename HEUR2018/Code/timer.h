#ifndef TIMER_H
#define TIMER_H

#include <sys/time.h>
/*class timer
{
 public:
  void init() { _start_time = clock(); } // postcondition: elapsed()==0
  double elapsed() const                  // return elapsed time in seconds
    { return  double(clock() - _start_time) / CLOCKS_PER_SEC; }

 private:
  clock_t _start_time;
};*/

/*class  timer {
private:
    time_t start;
public:
    void init(void) { time(&start); }
    double elapsed(void) const {
        time_t now;
        time(&now);
        return difftime(now, start);
    }
};
*/
class timer {
private:
    struct timeval tim;
    double tstart;
public:
    void init(void) {
        gettimeofday(&tim, NULL);
        //tstart = (double)tim.tv_sec + (double)tim.tv_usec/1000000.0;
        tstart = (double)(tim.tv_sec * 1000) + (tim.tv_usec / 1000);
    }
    double elapsed(void) {
        gettimeofday(&tim, NULL);
        double tcurrent = (double)(tim.tv_sec * 1000) + (tim.tv_usec / 1000);
        return (tcurrent - tstart)/1000; //time in seconds
    }

    double elapsedMiliSeconds(void) {
        gettimeofday(&tim, NULL);
        double tcurrent = (double)(tim.tv_sec * 1000) + (tim.tv_usec / 1000);
        return (tcurrent - tstart); //time in miliseconds
        
    }
    
};

#endif