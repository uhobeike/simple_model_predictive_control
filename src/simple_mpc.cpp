#include "ceres/ceres.h"
#include "glog/logging.h"

#include <numeric>

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;

const int N = 20;
const double dt = 0.1;

std::vector<double> path_xy;

struct RobotModel
{
  RobotModel(double dt, std::vector<double> path_xy) : dt_(dt), path_xy_(path_xy) {}

  template <typename T>
  bool operator()(const T * const v, const T * const w, T * residual) const
  {
    static std::vector<T> xs;
    static std::vector<T> ys;
    static std::vector<T> ths;
    static std::vector<T> costs;

    xs.push_back((T)0.0);
    ys.push_back((T)0.0);
    ths.push_back((T)0.00000000000000000000001);

    for (int i = 0; i < N; i++) {
      // clang-format off
      T x =
          xs[i] + 
            (v[i] / w[i]) * 
              (sin(ths[i] + w[i] * dt_) - sin(ths[i])) ;
      T y = 
          ys[i] +
            (v[i] / w[i]) * 
              ((T)-1.0 * cos(ths[i] + w[i] * dt_) + cos(ths[i]));
      T th = 
          ths[i] + 
            w[i] * dt_;
      // clang-format on

      T cost = sqrt(pow((path_xy_[i * 2] - x), 2) + pow((path_xy_[i * 2 + 1] - y), 2));

      xs.push_back(x);
      ys.push_back(y);
      ths.push_back(th);
      costs.push_back(cost);

      residual[i] = cost;
    }

    return true;
  }

private:
  double dt_;
  std::vector<double> path_xy_;
};

int main(int argc, char ** argv)
{
  google::InitGoogleLogging(argv[0]);

  path_xy.reserve(2 * N);
  for (int i = 0; i <= N; ++i) {
    double x = i;
    double y = std::sin(x);
    path_xy.push_back(x);
    path_xy.push_back(y);
  }

  std::vector<double> v_in(N, 0.0);
  std::vector<double> w_in(N, 0.0);
  auto v_out = v_in;
  auto w_out = w_in;

  Problem problem;
  problem.AddResidualBlock(
    new AutoDiffCostFunction<RobotModel, N, N, N>(new RobotModel(dt, path_xy)), nullptr,
    v_out.data(), w_out.data());
  const double lower_bound = 1.0e-10;
  for (int i = 0; i < N; ++i) {
    problem.SetParameterLowerBound(v_out.data(), i, lower_bound);
    problem.SetParameterLowerBound(w_out.data(), i, lower_bound);
  }

  Solver::Options options;
  options.max_num_iterations = 100;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;

  Solver::Summary summary;
  Solve(options, &problem, &summary);

  std::cout << summary.BriefReport() << "\n";

  std::cout << "Initial velocity: ";
  for (auto velocity : v_in) std::cout << velocity << ", ";
  std::cout << " omega: ";
  for (auto omega : w_in) std::cout << omega << ", ";
  std::cout << "\n";

  std::cout << "Final   velocity: ";
  for (auto velocity : v_out) std::cout << velocity << ", ";
  std::cout << " omega: ";
  for (auto omega : w_out) std::cout << omega << ", ";
  std::cout << "\n";

  return 0;
}