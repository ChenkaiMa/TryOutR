#include <RcppEigen.h>
#include <Rcpp.h>
using namespace Rcpp;
using namespace std;

// [[Rcpp::depends(RcppEigen)]]

static constexpr int separatorLengthLong = 3;
static constexpr int wordLengthShort = 10;
static constexpr int wordLengthMiddle = 20;
static constexpr int wordLengthLong = 25;
static constexpr int separatorLengthShort = 1;

static constexpr int iterationNumberLimitToPrint = 25;
static constexpr int iterationNumberIntervalToPrint = 10;

static constexpr int precisionForObjectiveValue = 3;
static constexpr int precisionForOrthogonalityViolation = 2;
static constexpr int precisionForTime = 3;

static constexpr int millisecondsToSeconds = 1000;

static constexpr int valueInDiagonal = 1;

static constexpr int defaultMaximumNumberOfIterations = 200;
static constexpr bool defaultIsVerbose = true;
static constexpr double defaultViolationTolerance = 1e-4;

static constexpr double initialOfvBest = -1e10;

static constexpr double slowPeriodRate = 0.15;
static constexpr double fastPeriodRate = 0.75;
static constexpr double changedRateLow = 0.01;
static constexpr double changedRateHigh = 0.05;

static constexpr int initialIterationNumber = 1;

static constexpr int rateForSigmaCurrent = 2;
static constexpr double differenceForLambda0 = 1e-4;
static constexpr double lowerLimitForViolation = 1e-7;

static constexpr int numbersOfSparseVersions = 100;
static constexpr int appliedTimeLimit = 20;
static constexpr int defaultTimeLimit = 720;
static constexpr int defaultSupportValue = 0;
static constexpr int defaultSupportSize = 1;
static constexpr int filledSupportValue = -1;
static constexpr int acceptableSupportValue = 1;
static constexpr int defaultCountdown = 100;
static constexpr int differenceFromMimValue = 1;

static constexpr double ofvPrecision = 1e-8;

Eigen::MatrixXd prob_data;
Eigen::MatrixXd prob_Sigma;
double ofv_best;
double violation_best;
double runtime;
double lambda_partial;
Eigen::VectorXd x_output;

int find_the_index(Rcpp::NumericVector aVector, double aNumber, Rcpp::NumericVector blocked)
{
  for (int i = 0; i < aVector.size(); i++) {
    if (fabs(aVector[i] - aNumber) < 1e-8) {
      int j = 0;
      for (; j < blocked.size(); j++) {
        if (fabs(blocked[i] - i) < 1e-8) {
          break;
        }
      }
      if (j == blocked.size()) {
        return i;
      }
    }
  }
  return 0;
}

double absoluteDouble(double aNumber) {
  if (aNumber >= 0) {
    return aNumber;
  }
  return -aNumber;
}

double evaluate(Eigen::VectorXd solution)
{
  return (solution.transpose() * prob_Sigma * solution)[0];
}

Rcpp::NumericVector selectperm2(Eigen::VectorXd x, int k)
{
  Rcpp::NumericVector numbers{};
  for (int i = 0; i < x.size(); i++) {
    numbers.push_back(absoluteDouble(x(i)));
  }

  Rcpp::NumericVector originalNumbers{};
  for (int i = 0; i < numbers.size(); i++) {
    originalNumbers.push_back(numbers[i]);
  }

  numbers.sort();
  Rcpp::NumericVector indexes{};
  for (int i = 0, j = numbers.size() - 1; i < k; i++, j--) {
    int index = find_the_index(originalNumbers, numbers[j], indexes);
    indexes.push_back(index);
  }
  return indexes;
}

Eigen::VectorXd Hk(Eigen::VectorXd origlist, int sparsity, Rcpp::NumericVector support)
{
  Eigen::VectorXd list = origlist;
  Eigen::VectorXd kparse = Eigen::VectorXd::Zero(list.size());
  int nbIndicesToKeep = 0;
  for (int s : support)
  {
    nbIndicesToKeep += s == 1;
  }

  double dummyValue = list(0) - 1;
  for (size_t i = 0; i < list.size(); i++)
  {
    if (list(i) - 1 < dummyValue) {
      dummyValue = list(i) - 1;
    }
  }
  for (int i = 0; i < list.size(); i++)
  {
    if (support[i] > -1)
    {
      list[i] = dummyValue;
    }
  }

  Rcpp::NumericVector newIndices = selectperm2(list, sparsity - nbIndicesToKeep);
  for (auto index : newIndices) {
    kparse[index] = origlist[index];
  }
  kparse = kparse / kparse.norm();
  return kparse;
}

Eigen::VectorXd eigSubset(Rcpp::NumericVector& support, int k, Eigen::VectorXd& beta0)
{

  Eigen::VectorXd beta = Hk(beta0, k, support);
  for (int i = 0; i < 100; i++)
  {
    beta = Hk(prob_Sigma * beta, k, support);
  }
  return beta;
}

void subset(int k, int timeLimit, Rcpp::NumericVector support, int countdown = 100)
{
  int n = prob_Sigma.rows();
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(prob_Sigma);
  int index;
  solver.eigenvalues().maxCoeff(&index);
  Eigen::VectorXd beta0 = solver.eigenvectors().col(index);
  if (support.size() == 1)
  {
    support[0] = -1;
    for (int i = 0; i < n - 1; i++) {
      support.push_back(-1);
    }
  }

  Eigen::VectorXd bestBeta = eigSubset(support, k, beta0);
  double bestObj = evaluate(bestBeta);
  time_t start = time(0);
  while (countdown > 0 && difftime(time(0), start) < timeLimit)
  {
    Eigen::VectorXd beta = Eigen::VectorXd::Zero(n);
    for (auto i = 0; i < beta.rows(); i++) {
      for (auto j = 0; j < beta.cols(); j++) {
        beta(i, j) = R::rnorm(0, 1);
      }
    }
    beta = beta / beta0.norm();
    beta = eigSubset(support, k, beta);
    double obj = evaluate(beta);
    if (obj > bestObj)
    {
      bestObj = obj;
      bestBeta = bestBeta;
      countdown = 100;
    }
    countdown--;
  }
  lambda_partial = bestObj;
  x_output = bestBeta;
  return;
}

double fnviolation(Eigen::MatrixXd x)
{
  double v = 0;
  Eigen::MatrixXd y = x.adjoint() * x;
  for (size_t i = 0; i < y.rows(); i++) {
    for (size_t j = 0; j < y.cols(); j++) {
      if (i == j) {
        v += double(fabs(y(i, j) - 1));
      }
      else {
        v += fabs(y(i, j));
      }
    }
  }
  return v;
}

// [[Rcpp::export]]
Eigen::MatrixXd cpp_findmultPCs_deflation(
    Eigen::MatrixXd Sigma,
    int r,
    Eigen::MatrixXd ks_input, // size r
    int numIters = 200,
    bool verbose = true,
    double violation_tolerance = 1e-4)
{
  Rcpp::NumericVector ks;
  for (int i = 0; i < ks_input.rows(); i++) {
    for (int j = 0; j < ks_input.cols(); j++) {
      ks.push_back(ks_input(i, j));
    }
  }
  int n = Sigma.rows();
  ofv_best = -1e10;
  violation_best = n;
  Eigen::MatrixXd x_best = Eigen::MatrixXd::Zero(n, r);
  Eigen::MatrixXd x_current = Eigen::MatrixXd::Zero(n, r);
  double ofv_prev = 0;
  double ofv_overall = 0;

  Eigen::VectorXd weights = Eigen::VectorXd::Zero(r);
  double theLambda = 0;
  double stepSize = 0;
  int slowPeriod = ceil(0.15 * numIters);
  int fastPeriod = ceil(0.75 * numIters);

  if (verbose)
  {
    Rcout << "---- Iterative deflation algorithm for sparse PCA with multiple PCs ---" << std::endl;
    Rcout << "Dimension: " << n << std::endl;
    Rcout << "Number of PCs: " << r << std::endl;
    Rcout << "Sparsity pattern: ";
    for (int t = 0; t < r; t++)
    {
      Rcout << " " << static_cast<int>(ks[t]);
    }
    Rcout << endl;
    Rcout << endl;


    Rcout.width(separatorLengthLong + wordLengthShort);
    Rcout << "Iteration |";

    Rcout.width(separatorLengthLong + wordLengthMiddle);
    Rcout << "Objective value |";


    Rcout.width(separatorLengthLong + wordLengthLong);
    Rcout << "Orthogonality Violation |";

    Rcout.width(separatorLengthShort + wordLengthShort);
    Rcout << "Time";
    Rcout << endl;
  }

  time_t start_time = time(0);
  for (int theIter = 1; theIter <= numIters; theIter++)
  {
    theLambda += stepSize;
    for (int t = 0; t < r; t++)
    {
      Eigen::MatrixXd sigma_current = Sigma;
      for (int s = 0; s < r; s++)
      {
        if (s != t)
        {
          sigma_current -= theLambda * weights[s] * x_current.col(s) * x_current.col(s).transpose();
        }
      }
      sigma_current = (sigma_current + sigma_current.transpose()) / 2;
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(sigma_current);
      double lambda0 = -solver.eigenvalues().minCoeff() + 1e-4;
      for (int i = 0; i < sigma_current.rows(); i++)
      {
        sigma_current(i, i) += lambda0;
      }
      solver = Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(sigma_current);
      prob_data = solver.operatorSqrt();
      prob_Sigma = sigma_current;

      Rcpp::NumericVector support;
      support.push_back(0);
      subset(ks[t], 20, support);

      x_current.col(t) = x_output;

      if (theIter == 1)
      {
        weights[t] = lambda_partial;
      }
    }
    ofv_prev = ofv_overall;
    ofv_overall = (x_current.transpose() * Sigma * x_current).trace();
    if (theIter == 1)
    {
      ofv_prev = ofv_overall;
    }

    double violation = fnviolation(x_current);
    if (1e-7 > violation) {
      violation = 1e-7;
    }
    stepSize = (theIter < fastPeriod ? 0.01 : 0.05) * (theIter < slowPeriod ? violation : ofv_overall / violation);
    if (verbose)
    {
      if (numIters <= 25 || theIter % 10 == 0)
      {
        double timespan = difftime(time(0), start_time);
        Rcout.width(separatorLengthShort + wordLengthShort);
        Rcout << theIter << " |";

        Rcout.width(separatorLengthShort + wordLengthMiddle);
        Rcout << setprecision(precisionForObjectiveValue) << ofv_overall / Sigma.trace() << " |";

        Rcout.width(separatorLengthShort + wordLengthLong);
        Rcout << scientific << setprecision(precisionForOrthogonalityViolation) << violation << " |" << defaultfloat;

        Rcout.width(separatorLengthShort + wordLengthShort);
        Rcout << fixed << setprecision(precisionForTime)
              << timespan << defaultfloat;
        Rcout << endl;
      }
    }

    if (violation < violation_tolerance || (theIter == numIters && ofv_best < 0))
    {
      double ofv_current = (x_current.transpose() * Sigma * x_current).trace();
      if (ofv_best < ofv_current)
      {
        x_best = x_current;
        ofv_best = ofv_current;
      }
    }

    if (fabs(ofv_prev - ofv_overall) < 1e-8 && violation < violation_tolerance)
    {
      if (ofv_best < 0) {
        x_best = x_current;
        ofv_best = (x_current.transpose() * Sigma * x_current).trace();
      }
      break;
    }
  }
  if (verbose)
  {
    printf("\n\n");
  }
  runtime = difftime(time(0), start_time);
  violation_best = fnviolation(x_best);
  ofv_best = (x_best.transpose() * Sigma * x_best).trace();
  return x_best;
}
