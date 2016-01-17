// OpenMPToy.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>
#include <boost\timer\timer.hpp>
#include <omp.h>
#include <Eigen\dense>
#include <chrono>

using namespace boost;
using namespace std;
using namespace std::chrono;
using namespace Eigen;


int num_steps = 100000000;
int _tmain(int argc, _TCHAR* argv[])
{
    int i;
    double x, step, sum = 0.0;
    step = 1. / (double)num_steps;
    timer::auto_cpu_timer *timer = new timer::auto_cpu_timer();
#pragma omp parallel for private(x) reduction(+:sum)
    for (i = 0; i < num_steps; i++)
    {
        x = (i + 0.5) * step;
        sum += 4.0 / (1.0 + x*x);
    }
    printf("PI = %.8f (sum = %.8f)\n", step*sum, sum);
    delete timer;

    timer = new timer::auto_cpu_timer;
    ArrayXf xx = (ArrayXf::LinSpaced((float)num_steps, 0.0, (float)num_steps - 1.0) + 0.5)*step;    
    ArrayXf yy =  4.0 / (1.0 + xx*xx);
    cout << yy.sum()*step << endl;
    delete timer;
    //printf("PI = %.8f (sum = %.8f)\n", step*yy, yy);

    timer = new timer::auto_cpu_timer;
    MatrixXf u(1000, 1000);
    u.setConstant(0.12);
    auto res = (u * u).colwise().sum();
    cout << res.sum() << endl;
    delete timer;

    //start = system_clock::now();
    //cout << "elapset msec =  " << duration_cast<milliseconds>(system_clock::now() - start).count() << endl;

	return 0;
}

