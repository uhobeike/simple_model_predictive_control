#include "ceres/ceres.h"
#include "glog/logging.h"

#include <sciplot/sciplot.hpp>

#include <chrono>
#include <thread>

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;

using namespace sciplot;

const int predictive_horizon_num = 40;
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
    std::vector<T> xs(predictive_horizon_num + 1, (T)0.0);
    std::vector<T> ys(predictive_horizon_num + 1, (T)0.0);
    std::vector<T> ths(predictive_horizon_num + 1, (T)0.0);

    for (int i = 0; i < predictive_horizon_num; i++) {
      // clang-format off
      T x =
            xs[i] + 
              v[i] * cos(ths[i]);
      T y = 
            ys[i] +
              v[i] * sin(ths[i]);
      T th = 
            ths[i] + 
              w[i] * dt_;
      // clang-format on

      T cost = pow((path_x_[i] - x), 2) + pow((path_y_[i] - y), 2);

      xs[i + 1] = x;
      ys[i + 1] = y;
      ths[i + 1] = th;

      residual[i] = cost;
    }

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

  std::vector<double> v_in(predictive_horizon_num, 0.0);
  std::vector<double> w_in(predictive_horizon_num, 0.0);
  auto v_out = v_in;
  auto w_out = w_in;

  Problem problem;
  problem.AddResidualBlock(
    new AutoDiffCostFunction<
      ObjectiveFunction, predictive_horizon_num, predictive_horizon_num, predictive_horizon_num>(
      new ObjectiveFunction(dt, path_x, path_y)),
    nullptr, v_out.data(), w_out.data());
  for (int i = 0; i < predictive_horizon_num; ++i) {
    problem.SetParameterLowerBound(v_out.data(), i, 0.0);
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

  for (int i = 0; i < predictive_horizon_num; i++) {
    // clang-format off
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

  std::string predictive_horizon_label = "Predictive Horizon";
  predictive_horizon_label =
    predictive_horizon_label + " N=" + std::to_string(predictive_horizon_num);
  plot.drawCurve(path_x, path_y).label("Path").lineColor("green");
  plot.drawCurveWithPoints(predictive_horizon_x, predictive_horizon_y)
    .label(predictive_horizon_label)
    .lineColor("red");

  Figure fig = {{plot}};
  Canvas canvas = {{fig}};
  canvas.size(1200, 800);
  canvas.show();

  return 0;
}