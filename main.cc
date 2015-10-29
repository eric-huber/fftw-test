#include "Array.h"
#include "fftw++.h"

#include <boost/program_options.hpp>

#include <cstdlib>
#include <chrono>
#include <iostream>
#include <fstream>
#include <iomanip>

using namespace std;
using namespace Array;
using namespace fftwpp;
using namespace chrono;

namespace po = boost::program_options;

typedef array1<double>  real_data;
typedef array1<Complex> imag_data;

string       _data_file_name    = "fft-data.txt";
string       _fft_file_name     = "fft-forward.txt";
string       _bak_file_name     = "fft-backward.txt";

unsigned int _fft_size      = 65536;
unsigned int _ifft_size     = _fft_size/2+1;

bool         _time          = false;
int          _batch         = 1000;
int          _count         = 100000;
float        _mean          = 0.5;
float        _std           = 0.2;
bool         _use_periodic  = false;

void randomize(real_data& data) {
    
    std::default_random_engine       generator(std::random_device{}());
    std::normal_distribution<float> distribution(_mean, _std);

    for (unsigned int i = 0; i < _fft_size; ++i) {
        data[i] = distribution(generator);
    }
}

void periodic(real_data& data) {
    for (unsigned int i = 0; i < _fft_size; ++i) {
        float t = i * .002;
        float amp = sin(M_PI * t);
        amp += sin(2 * M_PI * t);
        amp += sin(3 * M_PI * t); 
        data[i] = amp;
    }
}

void populate(real_data& data) {
    if (_use_periodic)
        periodic(data);
    else
        randomize(data);
}


double sqer(real_data& input, real_data& output) {
    
    // signal energy
    double se = 0.0;
    for (unsigned int i = 0; i < input.Size(); ++i) {
        se += pow(input[i], 2);
    }
    
    // quant error energy
    double qe = 0.0;
    for (unsigned int i = 0; i < input.Size(); ++i) {
        qe += pow(input[i] - output[i], 2);
    }    

    return 10 * log10(se / qe);
}


void write_real(real_data& data, string filename) {
    ofstream ofs;
    ofs.open(filename);
    ofs.precision(10);

    for (unsigned int i = 0; i < data.Size(); ++i) {
        ofs << data[i] << endl;
    }
    
    ofs.close();   
}

void write_imag(imag_data& data, string filename) {
    ofstream ofs;
    ofs.open(filename);
    ofs.precision(10);

    for (unsigned int i = 1; i < _ifft_size; i+=2) {
        double real = data[i].real();
        double imag = data[i].imag();
        double amp = sqrt(pow(real, 2) + pow(real, 2));
        double phase = atan2(imag, real);
        ofs << real << ", " << imag << ", " << amp << ", " << phase << endl;
    }
    
    ofs.close();
}

void summarize(long duration, double sqer) {
  
    int    count = _count * _batch;    
    double ave_dur = duration / (count * 2.0); // forward and reverse FFT
    double ave_sqer = sqer / count;
  
    cout.precision(8);
    cout << "Iterations: " << count << endl;
    cout << "Data size:  " << _fft_size << endl;
    cout << "Data type:  " << (_use_periodic ? "Periodic" : "Random") << endl;
    if (!_use_periodic) {
        cout << "Mean:       " << _mean << endl;
        cout << "Std Dev:    " << _std << endl;
    }
    cout << endl;
    if (_time) {   
        cout << "Time:       " << duration << " ns" << endl;
        cout << "Average:    " << ave_dur << " ns (" << (ave_dur / 1000.0) << " μs)" << endl;
        cout << "SQER:       " << sqer << endl;
        cout << "Ave SQER:   " << ave_sqer << endl;
    } else {
        cout << "SQER:       " << sqer << endl;
    }
}

void test_fft() {
    fftw::maxthreads = get_max_threads();

    size_t align = sizeof(Complex);
  
    real_data input(_fft_size, align);
    real_data output(_fft_size, align);
    imag_data imag(_ifft_size, align);
  
    rcfft1d forward(_fft_size, input, imag);
    crfft1d backward(_fft_size, imag, output);

    populate(input); 

    write_real(input, _data_file_name);

    forward.fft(input, imag);
    
    write_imag(imag, _fft_file_name);
    
    backward.fftNormalized(imag, output);
  
    write_real(output, _bak_file_name);
    
    summarize(0, sqer(input, output));
}

void time_fft() {
    
    fftw::maxthreads = get_max_threads();
    size_t align = sizeof(Complex);
  
    vector<real_data*>  input(_batch);  
    vector<real_data*>  output(_batch);
    vector<imag_data*>  imag(_batch);
    
    for (int i = 0; i < _batch; ++i) {
        real_data* in  = new real_data(_fft_size, align);
        real_data* out = new real_data(_fft_size, align);
        imag_data* im  = new imag_data(_ifft_size, align);
        
        input[i]  = in;
        output[i] = out;
        imag[i]   = im;
    }

    rcfft1d forward(_fft_size, *input[0], *imag[0]);
    crfft1d backward(_fft_size, *imag[0], *output[0]);
    
    double sqr = 0.0;
    nanoseconds duration(0);
    int last_percent = -1;
    
    for (int i = 0; i < _count; ++i) {

        for (int j = 0; j < _batch; ++j) {
            populate(*input[j]);
        }
    
        high_resolution_clock::time_point start = high_resolution_clock::now();
    
        for (int j = 0; j < _batch; ++j) {
            forward.fft(*input[j], *imag[j]);
            backward.fftNormalized(*imag[j], *output[j]);
        }
        
        high_resolution_clock::time_point stop = high_resolution_clock::now();
        duration += duration_cast<nanoseconds>(stop - start);
        
        for (int j = 0; j < _batch; ++j) {
            sqr += sqer(*input[j], *output[j]);
        }
        
        int percent = (int) ((double)i / _count * 100.0);
        if (percent != last_percent) {
            std::cout << "\r" << percent << " %";
            std::cout.flush();   
        }        
    }
    
    std::cout << "\r";
    summarize(duration.count(), sqr);    
}

int main(int ac, char* av[]) {
   try {
        
        po::options_description desc("Allowed options");
    
        desc.add_options()
        ("help,h",      "produce help message")
        ("time,t",      "Time the FFT operation")
        ("count,c",     po::value<int>(), "set the number of timed loops to perform")
        ("size,s",      po::value<int>(), "Set the size of the data buffer [8192]")
        ("mean,m",      po::value<float>(), "Set the range of the random data [25.0]")
        ("deviation,d", po::value<float>(), "Set the minimum value of the random data [0.0]")
        ("periodic,p",  "Use periodic instead of random data");

        po::variables_map vm;
        po::store(po::parse_command_line(ac, av, desc), vm);
        po::notify(vm);    

        if (vm.count("help")) {
            cout << desc << "\n";
            return 1;
        }
        
        if (vm.count("time")) {
            _time = true;
        }

        if (vm.count("count")) {
            _count = vm["count"].as<int>();
        }
    
        if (vm.count("size")) {
            _fft_size = vm["size"].as<int>();
        }
        
        if (vm.count("mean")) {
            _mean = vm["mean"].as<float>();
        }
        
        if (vm.count("deviation")) {
            _std = vm["deviation"].as<float>();
        }
        
        if (vm.count("periodic")) {
            _use_periodic = true;
        }

    } catch (exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    } catch (...) {
        cerr << "Unknown error" << endl;
        return 1;
    }

    _count = ceil((double)_count / _batch);

    if (_time)
        time_fft();
    else
        test_fft();
    
    return 0;
}
