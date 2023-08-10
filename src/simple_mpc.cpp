#include "simple_model_predictive_control/simple_mpc.hpp"

#include "ceres/ceres.h"
#include "glog/logging.h"

// using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::DynamicAutoDiffCostFunction;
using ceres::Problem;
using ceres::Solve;

constexpr int MAX_PREDICTIVE_HORIZON_NUM = 100;

SimpleMpc::SimpleMpc(
  std::vector<double> path_x, std::vector<double> path_y, std::vector<double> linear_velocity_in,
  std::vector<double> angular_velocity_in, std::vector<double> linear_velocity_out,
  std::vector<double> angular_velocity_out, const int predictive_horizon_num, const double dt,
  const double lower_bound_linear_velocity, const double lower_bound_angular_velocity,
  const double upper_bound_linear_velocity, const double upper_bound_angular_velocity)
: path_x_(path_x), path_y_(path_y), predictive_horizon_num_(predictive_horizon_num), dt_(dt)
{
  auto summary = optimization(
    linear_velocity_out, angular_velocity_out, lower_bound_linear_velocity,
    lower_bound_angular_velocity, upper_bound_linear_velocity, upper_bound_angular_velocity);
  auto predictive_horizon = getPredictiveHorizon(linear_velocity_out, angular_velocity_out);

  resultCout(
    summary, linear_velocity_in, angular_velocity_in, linear_velocity_out, angular_velocity_out);
  resultPlot(predictive_horizon);
}

Solver::Summary SimpleMpc::optimization(
  std::vector<double> & v_out, std::vector<double> & w_out,
  const double lower_bound_linear_velocity, const double lower_bound_angular_velocity,
  const double upper_bound_linear_velocity, const double upper_bound_angular_velocity)
{
  auto * cost_function =
    new ceres::DynamicAutoDiffCostFunction<ObjectiveFunction, MAX_PREDICTIVE_HORIZON_NUM>(
      new ObjectiveFunction(path_x_, path_y_, dt_, predictive_horizon_num_));

  cost_function->SetNumResiduals(predictive_horizon_num_);
  cost_function->AddParameterBlock(predictive_horizon_num_);
  cost_function->AddParameterBlock(predictive_horizon_num_);

  Problem problem;
  problem.AddResidualBlock(cost_function, nullptr, v_out.data(), w_out.data());

  for (int i = 0; i < predictive_horizon_num_; ++i) {
    problem.SetParameterLowerBound(v_out.data(), i, lower_bound_linear_velocity);
    problem.SetParameterLowerBound(w_out.data(), i, lower_bound_angular_velocity);
    problem.SetParameterUpperBound(v_out.data(), i, upper_bound_linear_velocity);
    problem.SetParameterUpperBound(w_out.data(), i, upper_bound_angular_velocity);
  }

  Solver::Options options;
  options.max_num_iterations = 100;
  options.linear_solver_type = ceres::DENSE_QR;
  options.minimizer_progress_to_stdout = true;

  Solver::Summary summary;
  Solve(options, &problem, &summary);

  return summary;
}

std::pair<std::vector<double>, std::vector<double>> SimpleMpc::getPredictiveHorizon(
  std::vector<double> v_out, std::vector<double> w_out)
{
  std::vector<double> predictive_horizon_x;
  std::vector<double> predictive_horizon_y;
  std::vector<double> ths;

  predictive_horizon_x.push_back((double)0.0);
  predictive_horizon_y.push_back((double)0.0);
  ths.push_back((double)0.0);

  for (int i = 0; i < predictive_horizon_num_; i++) {
    // clang-format off
    double  x =
          predictive_horizon_x[i] + 
            v_out[i] * cos(ths[i]) * dt_;
    double  y = 
          predictive_horizon_y[i] +
            v_out[i] * sin(ths[i]) * dt_;
    double  th = 
          ths[i] + 
            w_out[i] * dt_;
    // clang-format on

    predictive_horizon_x.push_back(x);
    predictive_horizon_y.push_back(y);
    ths.push_back(th);
  }

  return std::make_pair(predictive_horizon_x, predictive_horizon_y);
}

void SimpleMpc::resultCout(
  Solver::Summary summary, std::vector<double> v_in, std::vector<double> w_in,
  std::vector<double> v_out, std::vector<double> w_out)
{
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
}

void SimpleMpc::resultPlot(std::pair<std::vector<double>, std::vector<double>> predictive_horizon)
{
  sciplot::Plot2D plot;
  plot.fontName("Palatino");
  plot.xlabel("x").fontSize(20);
  plot.ylabel("y").fontSize(20);
  plot.legend().atTop().fontSize(20).displayHorizontal().displayExpandWidthBy(2);
  plot.grid().show();

  std::string predictive_horizon_label = "Predictive Horizon";
  predictive_horizon_label =
    predictive_horizon_label + " N=" + std::to_string(predictive_horizon_num_);
  plot.drawCurve(path_x_, path_y_).label("Path").lineColor("green");
  plot.drawCurveWithPoints(predictive_horizon.first, predictive_horizon.second)
    .label(predictive_horizon_label)
    .lineColor("red");

  sciplot::Figure fig = {{plot}};
  sciplot::Canvas canvas = {{fig}};
  canvas.size(1200, 800);
  canvas.show();
}

void createPath(std::vector<double> & path_x, std::vector<double> & path_y)
{
  for (int i = 0; i <= 100; ++i) {
    double x = static_cast<double>(i);
    path_x.push_back(x / 10);
    path_y.push_back(sin(x / 10));
  }
}

void initVelocitys(
  std::vector<std::vector<double> *> & vectors, const double predictive_horizon_num)
{
  for (auto vec : vectors) vec->assign(predictive_horizon_num, 0.0);
}

int main(int argc, char ** argv)
{
  google::InitGoogleLogging(argv[0]);

  // MPC Parameters
  constexpr double dt = 1;
  constexpr int predictive_horizon_num = 40;

  constexpr double lower_bound_linear_velocity = 0.0;
  constexpr double lower_bound_angular_velocity = -M_PI;
  constexpr double upper_bound_linear_velocity = 1.0;
  constexpr double upper_bound_angular_velocity = M_PI;

  std::vector<double> path_x, path_y;
  createPath(path_x, path_y);

  std::vector<double> v_in, w_in, v_out, w_out;
  std::vector<std::vector<double> *> vectors = {&v_in, &w_in, &v_out, &w_out};
  initVelocitys(vectors, predictive_horizon_num);

  auto mpc = std::make_shared<SimpleMpc>(
    path_x, path_y, v_in, w_in, v_out, w_out, predictive_horizon_num, dt,
    lower_bound_linear_velocity, lower_bound_angular_velocity, upper_bound_linear_velocity,
    upper_bound_angular_velocity);

  return 0;
}