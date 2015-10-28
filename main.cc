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

int          _data_count    = 1;
bool         _time          = false;
int          _count         = 100000;
float        _mean          = 0.5;
float        _std           = 0.2;
float        _invert        = false;
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

void test_fft() {
    fftw::maxthreads = get_max_threads();

    size_t align = sizeof(Complex);
  
    array1<double> real(_fft_size, align);
    array1<Complex> imag(_ifft_size, align);
  
    rcfft1d forward(_fft_size, real, imag);
    crfft1d backward(_fft_size, imag, real);

    populate(real);  

    write_real(real, _data_file_name);

    forward.fft(real, imag);
    
    write_imag(imag, _fft_file_name);
    
    backward.fftNormalized(imag, real);
  
    write_real(real, _bak_file_name);
}

int main(int ac, char* av[]) {
   try {
        
        po::options_description desc("Allowed options");
    
        desc.add_options()
        ("help,h",      "produce help message")
        ("time,t",      "Time the FFT operation")
        ("invert,i",    "Perform timings on both the  FFT and inverse FFT")
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
            _data_count = 1000;
        }
        
        if (vm.count("invert")) {
            _invert = true;
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

    test_fft();
    
    return 0;
}
