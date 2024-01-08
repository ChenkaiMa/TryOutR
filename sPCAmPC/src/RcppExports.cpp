// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppEigen.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// cpp_findmultPCs_deflation
Eigen::MatrixXd cpp_findmultPCs_deflation(Eigen::MatrixXd Sigma, int r, Eigen::MatrixXd ks_input, int numIters, bool verbose, double violation_tolerance);
RcppExport SEXP _sPCAmPC_cpp_findmultPCs_deflation(SEXP SigmaSEXP, SEXP rSEXP, SEXP ks_inputSEXP, SEXP numItersSEXP, SEXP verboseSEXP, SEXP violation_toleranceSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type Sigma(SigmaSEXP);
    Rcpp::traits::input_parameter< int >::type r(rSEXP);
    Rcpp::traits::input_parameter< Eigen::MatrixXd >::type ks_input(ks_inputSEXP);
    Rcpp::traits::input_parameter< int >::type numIters(numItersSEXP);
    Rcpp::traits::input_parameter< bool >::type verbose(verboseSEXP);
    Rcpp::traits::input_parameter< double >::type violation_tolerance(violation_toleranceSEXP);
    rcpp_result_gen = Rcpp::wrap(cpp_findmultPCs_deflation(Sigma, r, ks_input, numIters, verbose, violation_tolerance));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_sPCAmPC_cpp_findmultPCs_deflation", (DL_FUNC) &_sPCAmPC_cpp_findmultPCs_deflation, 6},
    {NULL, NULL, 0}
};

RcppExport void R_init_sPCAmPC(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
