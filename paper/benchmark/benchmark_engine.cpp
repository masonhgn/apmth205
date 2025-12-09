

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <random>
#include <fstream>
#include <iomanip>

//ez calculation
constexpr double INV_SQRT_2PI = 0.3989422804014327;
constexpr double M_SQRT1_2_VAL = 0.7071067811865476;
constexpr double EPSILON_PRICE = 1e-8;
constexpr double EPSILON_VEGA  = 1e-12;

struct OptionParams {
    double price; double S; double K; double T; double r; bool is_call;
};

//helpers
double norm_cdf(double x) { return 0.5 * std::erfc(-x * M_SQRT1_2_VAL); }
double norm_pdf(double x) { return INV_SQRT_2PI * std::exp(-0.5 * x * x); }

double bs_price(double sigma, const OptionParams& p) {
    double d1 = (std::log(p.S/p.K) + (p.r + 0.5*sigma*sigma)*p.T) / (sigma*std::sqrt(p.T));
    double d2 = d1 - sigma*std::sqrt(p.T);
    if(p.is_call) return p.S * norm_cdf(d1) - p.K * std::exp(-p.r*p.T) * norm_cdf(d2);
    else          return p.K * std::exp(-p.r*p.T) * norm_cdf(-d2) - p.S * norm_cdf(-d1);
}

//standard hybrid solver
int solve_standard(const OptionParams& p, double& sigma_out, std::vector<double>* trace = nullptr) {
    double vol_min = 1e-5, vol_max = 5.0;
    double sigma = 0.5; //could be a bad guess
    int iter = 0;
    
    for (; iter < 100; ++iter) {
        double model = bs_price(sigma, p);
        double error = model - p.price;
        
        if (trace) trace->push_back(std::abs(error));
        if (std::abs(error) < EPSILON_PRICE) break;

        //bisection bracket 
        if (error > 0) vol_max = sigma; else vol_min = sigma;

        //derivatives
        double d1 = (std::log(p.S/p.K) + (p.r + 0.5*sigma*sigma)*p.T) / (sigma*std::sqrt(p.T));
        double vega = p.S * std::sqrt(p.T) * norm_pdf(d1);


        if (vega < EPSILON_VEGA) {
            sigma = (vol_min + vol_max) * 0.5;
        } else {
            double sigma_new = sigma - error / vega;
            // Bracket Guard
            if (sigma_new <= vol_min || sigma_new >= vol_max)
                sigma = (vol_min + vol_max) * 0.5;
            else
                sigma = sigma_new;
        }
    }
    sigma_out = sigma;
    return iter;
}

//gcmh
double get_initialization(const OptionParams& p) {
    double X = p.K * std::exp(-p.r * p.T);
    double C = p.is_call ? p.price : p.price + p.S - X; 
    double diff = p.S - X;
    double L = C - diff / 2.0;
    double discriminant = L*L - (diff*diff)/M_PI;

    //asymptotic fallback
    if (discriminant < 0.0) 
        return std::sqrt(2.0 * std::abs(std::log(p.S / p.K)) / p.T);

    //CM
    double root = std::sqrt(discriminant);
    return (std::sqrt(2.0 * M_PI) / ((p.S + X) * std::sqrt(p.T))) * (L + root);
}

int solve_gcmh(const OptionParams& p, double& sigma_out, std::vector<double>* trace = nullptr) {
    double vol_min = 1e-5, vol_max = 5.0;
    
    //above
    double sigma = get_initialization(p);
    sigma = std::max(vol_min, std::min(sigma, vol_max));

    int iter = 0;
    for (; iter < 100; ++iter) {
        double d1 = (std::log(p.S/p.K) + (p.r + 0.5*sigma*sigma)*p.T) / (sigma*std::sqrt(p.T));
        double d2 = d1 - sigma*std::sqrt(p.T);
        double model = p.is_call ? p.S*norm_cdf(d1) - p.K*std::exp(-p.r*p.T)*norm_cdf(d2)
                                 : p.K*std::exp(-p.r*p.T)*norm_cdf(-d2) - p.S*norm_cdf(-d1);
        
        double error = model - p.price;
        if (trace) trace->push_back(std::abs(error));
        if (std::abs(error) < EPSILON_PRICE) break;

        //update brackets
        if (error > 0) vol_max = sigma; else vol_min = sigma;

        double vega = p.S * std::sqrt(p.T) * norm_pdf(d1);

        //vega guard
        if (vega < EPSILON_VEGA) {
            sigma = (vol_min + vol_max) * 0.5;
            continue;
        }

        //halley itr
        double vomma = vega * d1 * d2 / sigma;
        double denom = 2*vega*vega - error*vomma;
        
        double sigma_new;
        if (std::abs(denom) < 1e-10) sigma_new = sigma - error/vega; // Newton fallback
        else sigma_new = sigma - (2*error*vega)/denom;

        //bracket
        if (sigma_new <= vol_min || sigma_new >= vol_max)
            sigma = (vol_min + vol_max) * 0.5;
        else
            sigma = sigma_new;
    }
    sigma_out = sigma;
    return iter;
}










void run_convergence_trace() {
    std::cout << "[1/4] Running Convergence Trace..." << std::endl;
    // deep OTM case: S=100, K=140, T=0.1, Vol=0.3
    OptionParams p = {0, 100.0, 140.0, 0.1, 0.0, true};
    p.price = bs_price(0.3, p);

    std::vector<double> t_std, t_gcmh;
    double dummy;
    solve_standard(p, dummy, &t_std);
    solve_gcmh(p, dummy, &t_gcmh);

    std::ofstream f("data_trace.csv");
    f << "Iteration,Err_Standard,Err_GCMH\n";
    size_t n = std::max(t_std.size(), t_gcmh.size());
    for(size_t i=0; i<n; ++i) {
        f << (i+1) << ",";
        if(i < t_std.size()) f << t_std[i] << ","; else f << ",";
        if(i < t_gcmh.size()) f << t_gcmh[i]; else f << "";
        f << "\n";
    }
    f.close();
}




void run_heatmap_scan() {
    std::cout << "[2/4] Running Heatmap Scan..." << std::endl;
    std::ofstream f("data_heatmap.csv");
    f << "Moneyness,Time,Iter_Standard,Iter_GCMH\n";
    
    double dummy;
    for(double m=0.5; m<=2.0; m+=0.05) {
        for(double t=0.1; t<=2.0; t+=0.1) {
            OptionParams p = {0, 100.0, 100.0*m, t, 0.05, true};
            p.price = bs_price(0.3, p); // Solve for 30% vol
            int i_std = solve_standard(p, dummy);
            int i_new = solve_gcmh(p, dummy);
            f << m << "," << t << "," << i_std << "," << i_new << "\n";
        }
    }
    f.close();
}

void run_efficiency_benchmark() {
    std::cout << "[3/4] Running Wall-Clock Efficiency..." << std::endl;
    const int N = 1000000;
    std::vector<OptionParams> universe;
    universe.reserve(N);
    
    std::mt19937 gen(12345);
    std::uniform_real_distribution<> K_dist(50, 150);
    std::uniform_real_distribution<> T_dist(0.1, 2.0);

    for(int i=0; i<N; ++i) {
        OptionParams p = {0, 100.0, K_dist(gen), T_dist(gen), 0.05, true};
        p.price = bs_price(0.3, p);
        universe.push_back(p);
    }

    double dummy;
    volatile double sink = 0;

    auto t1 = std::chrono::high_resolution_clock::now();
    for(const auto& p : universe) { solve_standard(p, dummy); sink += dummy; }
    auto t2 = std::chrono::high_resolution_clock::now();
    for(const auto& p : universe) { solve_gcmh(p, dummy); sink += dummy; }
    auto t3 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> d_std = t2 - t1;
    std::chrono::duration<double> d_new = t3 - t2;

    std::ofstream f("data_efficiency.csv");
    f << "Method,TimeSec,SolvesPerSec\n";
    f << "Standard," << d_std.count() << "," << N/d_std.count() << "\n";
    f << "GCM-H," << d_new.count() << "," << N/d_new.count() << "\n";
    f.close();
}

void run_accuracy_generation() {
    std::cout << "[4/4] Generating Accuracy Verification Data..." << std::endl;
    //export inputs and result 
    //python script compares to jackel LBR as ground truth
    std::ofstream f("data_accuracy.csv");
    f << "S,K,T,r,Price,Solved_GCMH\n";

    // Scan volatilities
    for(double v=0.05; v<=1.5; v+=0.05) {
        OptionParams p = {0, 100.0, 110.0, 1.0, 0.05, true};
        p.price = bs_price(v, p);
        
        double solved_sigma;
        solve_gcmh(p, solved_sigma);
        
        f << p.S << "," << p.K << "," << p.T << "," << p.r << "," 
          << std::setprecision(16) << p.price << "," << solved_sigma << "\n";
    }
    f.close();
}

int main() {
    run_convergence_trace();
    run_heatmap_scan();
    run_efficiency_benchmark();
    run_accuracy_generation();
    std::cout << "Done. CSV files generated." << std::endl;
    return 0;
}