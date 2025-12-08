#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <fstream>
#include <iomanip>

//constants for common terms
constexpr double INV_SQRT_2PI = 0.3989422804014327;
constexpr double M_SQRT1_2_VAL = 0.7071067811865476;
constexpr double EPSILON_PRICE = 1e-8;
constexpr double EPSILON_VEGA  = 1e-12;

struct OptionParams { //struct for opti on
    double price; double S; double K; double T; double r; bool is_call;
};

//kernels
double norm_cdf(double x) { return 0.5 * std::erfc(-x * M_SQRT1_2_VAL); }
double norm_pdf(double x) { return INV_SQRT_2PI * std::exp(-0.5 * x * x); }

double bs_price(double sigma, const OptionParams& p) {
    double d1 = (std::log(p.S/p.K) + (p.r + 0.5*sigma*sigma)*p.T) / (sigma*std::sqrt(p.T));
    double d2 = d1 - sigma*std::sqrt(p.T);
    if(p.is_call) return p.S * norm_cdf(d1) - p.K * std::exp(-p.r*p.T) * norm_cdf(d2);
    else          return p.K * std::exp(-p.r*p.T) * norm_cdf(-d2) - p.S * norm_cdf(-d1);
}

//solver: standard hybrid, no smart init
int solve_hybrid_standard(const OptionParams& p, std::vector<double>* error_trace = nullptr) {
    double vol_min = 1e-5, vol_max = 5.0;
    double sigma = 0.5; //naive Guess
    int iter = 0;
    for (; iter < 50; ++iter) {
        double model = bs_price(sigma, p);
        double error = model - p.price;
        if (error_trace) error_trace->push_back(std::abs(error));
        
        if (std::abs(error) < EPSILON_PRICE) break;
        if (error > 0) vol_max = sigma; else vol_min = sigma;

        double d1 = (std::log(p.S/p.K) + (p.r + 0.5*sigma*sigma)*p.T) / (sigma*std::sqrt(p.T));
        double vega = p.S * std::sqrt(p.T) * norm_pdf(d1);

        if (vega < 1e-8) { //gradient vanished
            sigma = (vol_min + vol_max) * 0.5; //bisection
        } else {
            double sigma_new = sigma - error / vega; // Newton
            if (sigma_new <= vol_min || sigma_new >= vol_max) 
                sigma = (vol_min + vol_max) * 0.5; //bracket Guard
            else 
                sigma = sigma_new;
        }
    }
    return iter;
}






//SOLVER: GCM-H
double get_gcmh_init(const OptionParams& p) {
    double X = p.K * std::exp(-p.r * p.T);
    double C = p.is_call ? p.price : p.price + p.S - X;
    double diff = p.S - X; 
    double term_linear = C - diff / 2.0;
    double discriminant = term_linear * term_linear - (diff * diff) / M_PI;

    if (discriminant < 0.0) return std::sqrt(2.0 * std::abs(std::log(p.S / p.K)) / p.T);
    double root_term = std::sqrt(discriminant);
    return (std::sqrt(2.0 * M_PI) / ((p.S + X) * std::sqrt(p.T))) * (term_linear + root_term);
}

int solve_gcmh(const OptionParams& p, std::vector<double>* error_trace = nullptr) {
    double vol_min = 1e-5, vol_max = 5.0;
    double sigma = get_gcmh_init(p);
    sigma = std::max(vol_min, std::min(sigma, vol_max));

    int iter = 0;
    for (; iter < 50; ++iter) {
        double d1 = (std::log(p.S/p.K) + (p.r + 0.5*sigma*sigma)*p.T) / (sigma*std::sqrt(p.T));
        double d2 = d1 - sigma*std::sqrt(p.T);
        double model = (p.is_call) ? p.S*norm_cdf(d1)-p.K*std::exp(-p.r*p.T)*norm_cdf(d2) 
                                   : p.K*std::exp(-p.r*p.T)*norm_cdf(-d2)-p.S*norm_cdf(-d1);
        
        double error = model - p.price;
        if (error_trace) error_trace->push_back(std::abs(error));
        
        if (std::abs(error) < EPSILON_PRICE) break;
        if (error > 0) vol_max = sigma; else vol_min = sigma;

        double vega = p.S * std::sqrt(p.T) * norm_pdf(d1);
        if (vega < EPSILON_VEGA) {
            sigma = (vol_min + vol_max) * 0.5; 
            continue;
        }
        
        double vomma = vega * d1 * d2 / sigma;
        double denom = 2*vega*vega - error*vomma;
        double sigma_new = (std::abs(denom)<1e-10) ? (sigma - error/vega) : (sigma - (2*error*vega)/denom);
        
        if (sigma_new <= vol_min || sigma_new >= vol_max) sigma = (vol_min + vol_max) * 0.5;
        else sigma = sigma_new;
    }
    return iter;
}

int main() {
    // 1. GENERATE CONVERGENCE TRACE (Deep OTM)
    // S=100, K=140, T=0.1, Vol=0.3 -> Call Price ~ 0.0003
    OptionParams p_stress = {0.0003, 100.0, 140.0, 0.1, 0.0, true};
    
    std::vector<double> trace_old, trace_new;
    solve_hybrid_standard(p_stress, &trace_old);
    solve_gcmh(p_stress, &trace_new);

    std::ofstream f1("convergence_trace.csv");
    f1 << "Iteration,Standard_Error,GCMH_Error\n";
    size_t max_len = std::max(trace_old.size(), trace_new.size());
    for(size_t i=0; i<max_len; ++i) {
        f1 << (i+1) << ",";
        if(i < trace_old.size()) f1 << trace_old[i] << ","; else f1 << ",";
        if(i < trace_new.size()) f1 << trace_new[i]; else f1 << "";
        f1 << "\n";
    }
    f1.close();
    std::cout << "Generated convergence_trace.csv" << std::endl;

    // 2. GENERATE HEATMAP DATA
    std::ofstream f2("heatmap_data.csv");
    f2 << "Moneyness,Time,Iter_Standard,Iter_GCMH\n";
    
    // Grid: Moneyness 0.5 to 1.5, Time 0.1 to 2.0
    for(double m = 0.5; m <= 1.5; m += 0.05) {
        for(double t = 0.1; t <= 2.0; t += 0.1) {
            double S = 100.0;
            double K = S * m; // Moneyness definition K/S usually, here using K = S*m
            double target_vol = 0.3;
            // Calculate "market price" for this target vol
            OptionParams p_grid = {0, S, K, t, 0.0, true};
            p_grid.price = bs_price(target_vol, p_grid);
            
            int it_old = solve_hybrid_standard(p_grid);
            int it_new = solve_gcmh(p_grid);
            f2 << m << "," << t << "," << it_old << "," << it_new << "\n";
        }
    }
    f2.close();
    std::cout << "Generated heatmap_data.csv" << std::endl;

    return 0;
}