#include "sciplot/sciplot.hpp"

#include <ceres/ceres.h>
#include <glog/logging.h>

using ceres::Solver;

class SimpleMpc
{
public:
  SimpleMpc(
    std::vector<double> path_x, std::vector<double> path_y, std::vector<double> linear_velocity_in,
    std::vector<double> angular_velocity_in, std::vector<double> linear_velocity_out,
    std::vector<double> angular_velocity_out, const int predictive_horizon_num, const double dt,
    const double lower_bound_linear_velocity, const double lower_bound_angular_velocity,
    const double upper_bound_linear_velocity, const double upper_bound_angular_velocity);

protected:
  Solver::Summary optimization(
    std::vector<double> & v_out, std::vector<double> & w_out,
    const double lower_bound_linear_velocity, const double lower_bound_angular_velocity,
    const double upper_bound_linear_velocity, const double upper_bound_angular_velocity);

  std::pair<std::vector<double>, std::vector<double>> getPredictiveHorizon(
    std::vector<double> v_out, std::vector<double> w_out);

  void resultCout(
    Solver::Summary summary, std::vector<double> v_in, std::vector<double> w_in,
    std::vector<double> v_out, std::vector<double> w_out);
  void resultPlot(std::pair<std::vector<double>, std::vector<double>> predictive_horizon);

private:
  std::vector<double> path_x_, path_y_;

  const int predictive_horizon_num_;
  const double dt_;
};

struct ObjectiveFunction
{
  ObjectiveFunction(
    std::vector<double> path_x, std::vector<double> path_y, double dt, int predictive_horizon_num)
  : path_x_(path_x), path_y_(path_y), dt_(dt), predictive_horizon_num_(predictive_horizon_num)
  {
  }

  template <typename T>
  bool operator()(T const * const * parameters, T * residual) const
  {
    std::vector<T> xs(predictive_horizon_num_ + 1, (T)0.0);
    std::vector<T> ys(predictive_horizon_num_ + 1, (T)0.0);
    std::vector<T> ths(predictive_horizon_num_ + 1, (T)0.0);

    for (int i = 0; i < predictive_horizon_num_; i++) {
      // clang-format off
      T x =
            xs[i] + 
              parameters[0][i] * cos(ths[i]) * dt_;
      T y = 
            ys[i] +
              parameters[0][i] * sin(ths[i]) * dt_;
      T th = 
            ths[i] + 
              parameters[1][i] * dt_;
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
  std::vector<double> path_x_;
  std::vector<double> path_y_;
  double dt_;
  int predictive_horizon_num_;
};