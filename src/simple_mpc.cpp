#include "ceres/ceres.h"
#include "glog/logging.h"

#include <sciplot/sciplot.hpp>

#include <chrono>
#include <numeric>
#include <thread>

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;

using namespace sciplot;

const int N = 40;
const double dt = 1;

std::vector<double> path_x;
std::vector<double> path_y;

struct ObjectiveFunction
{
  ObjectiveFunction(double dt, std::vector<double> path_x, std::vector<double> path_y)
  : dt_(dt), path_x_(path_x), path_y_(path_y)
  {
  }

  template <typename T>
  bool operator()(const T * const v, const T * const w, T * residual) const
  {
    std::vector<T> xs(N + 1, (T)0.0);
    std::vector<T> ys(N + 1, (T)0.0);
    std::vector<T> ths(N + 1, (T)0.0);

    for (int i = 0; i < N; i++) {
      // clang-format off
      // if (w[i] >= (T)1.0e-100) {
        // x =
        //     xs[i] + 
        //       (v[i] / w[i]) * 
        //         (sin(ths[i] + w[i] * dt_) - sin(ths[i]));
        // y = 
        //     ys[i] +
        //       (v[i] / w[i]) * 
        //         ((T)-1.0 * cos(ths[i] + w[i] * dt_) + cos(ths[i]));
        // th = 
        //     ths[i] + 
        //       w[i] * dt_;
      // }
      // if(w[i] <= (T)1.0e-100) {
      T  x =
            xs[i] + 
              v[i] * cos(ths[i]);
      T  y = 
            ys[i] +
              v[i] * sin(ths[i]);
      T  th = 
            ths[i] + 
              w[i] * dt_;
      // }

      // clang-format on

      T cost = pow((path_x_[i] - x), 2) + pow((path_y_[i] - y), 2);

      xs[i + 1] = x;
      ys[i + 1] = y;
      ths[i + 1] = th;

      // cost += cost;

      residual[i] = cost;
    }

    // residual[0] = accumulate(costs.begin(), costs.end(), (T)0.0);
    // residual[0] = cost;

    return true;
  }

private:
  double dt_;
  std::vector<double> path_x_;
  std::vector<double> path_y_;
};

int main(int argc, char ** argv)
{
  google::InitGoogleLogging(argv[0]);

  for (int i = 0; i <= 100; ++i) {
    double x = static_cast<double>(i);
    path_x.push_back(x / 10);
    path_y.push_back(sin(x / 10));
  }

  std::vector<double> v_in(N, 0.0);
  std::vector<double> w_in(N, 0.0);
  auto v_out = v_in;
  auto w_out = w_in;

  Problem problem;
  problem.AddResidualBlock(
    new AutoDiffCostFunction<ObjectiveFunction, N, N, N>(new ObjectiveFunction(dt, path_x, path_y)),
    nullptr, v_out.data(), w_out.data());
  for (int i = 0; i < N; ++i) {
    problem.SetParameterLowerBound(v_out.data(), i, 1.0e-10);
    problem.SetParameterLowerBound(w_out.data(), i, -3.14);
    problem.SetParameterUpperBound(v_out.data(), i, 1.0);
    problem.SetParameterUpperBound(w_out.data(), i, 3.14);
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

  std::vector<double> predictive_horizon_x;
  std::vector<double> predictive_horizon_y;
  std::vector<double> ths;

  predictive_horizon_x.push_back((double)0.0);
  predictive_horizon_y.push_back((double)0.0);
  ths.push_back((double)0.0);

  for (int i = 0; i < N; i++) {
    // clang-format off
      // double x =
      //     predictive_horizon_x[i] + 
      //       (v_out[i] / w_out[i]) * 
      //         (sin(ths[i] + w_out[i] * dt) - sin(ths[i])) ;
      // double y = 
      //     predictive_horizon_y[i] +
      //       (v_out[i] / w_out[i]) * 
      //         (-1.0 * cos(ths[i] + w_out[i] * dt) + cos(ths[i]));
      // double th = 
      //     ths[i] + 
      //       w_out[i] * dt;
      double  x =
            predictive_horizon_x[i] + 
              v_out[i] * cos(ths[i]);
      double  y = 
            predictive_horizon_y[i] +
              v_out[i] * sin(ths[i]);
      double  th = 
            ths[i] + 
              w_out[i] * dt;
    // clang-format on

    predictive_horizon_x.push_back(x);
    predictive_horizon_y.push_back(y);
    ths.push_back(th);
  }

  Plot2D plot;
  plot.fontName("Palatino");
  plot.xlabel("x").fontSize(20);
  plot.ylabel("y").fontSize(20);
  plot.legend().atTop().fontSize(20).displayHorizontal().displayExpandWidthBy(2);
  plot.grid().show();

  plot.drawCurve(path_x, path_y).label("Path").lineColor("green");
  plot.drawCurve(predictive_horizon_x, predictive_horizon_y)
    .label("Predictive Horizon")
    .lineColor("red");

  Figure fig = {{plot}};
  Canvas canvas = {{fig}};
  canvas.size(1000, 1000);
  canvas.show();

  return 0;
}