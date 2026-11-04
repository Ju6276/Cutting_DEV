// Copyright 2019 FZI Research Center for Information Technology.
// SPDX-License-Identifier: BSD-3-Clause; see licenses/FZI-BSD-3-Clause.txt.
// Generic inertia construction follows ForwardDynamicsSolver.cpp at f4359b966.
#include <kdl_parser/kdl_parser.hpp>
#include <kdl/chaindynparam.hpp>
#include <kdl/chainjnttojacsolver.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <iomanip>
#include <algorithm>
#include <Eigen/LU>
#include <iostream>

int main(int argc, char** argv) {
  if (argc != 5 && argc != 7) return 2;
  KDL::Tree tree;
  if (!kdl_parser::treeFromFile(argv[1], tree)) return 3;
  KDL::Chain chain;
  if (!tree.getChain(argv[2], argv[3], chain)) return 4;
  const double mass = std::stod(argv[4]);
  for (auto& segment : chain.segments) {
    segment.setInertia(segment.getJoint().getType() == KDL::Joint::None
        ? KDL::RigidBodyInertia::Zero()
        : KDL::RigidBodyInertia(mass, KDL::Vector::Zero(), KDL::RotationalInertia(1e-6, 1e-6, 1e-6)));
  }
  chain.segments.back().setInertia(KDL::RigidBodyInertia(1., KDL::Vector::Zero(), KDL::RotationalInertia(1., 1., 1.)));
  KDL::ChainDynParam dynamics(chain, KDL::Vector::Zero());
  KDL::ChainJntToJacSolver jacobian(chain);
  KDL::ChainFkSolverPos_recursive forward(chain);
  KDL::JntArray q(chain.getNrOfJoints());
  KDL::JntSpaceInertiaMatrix h(q.rows());
  KDL::Jacobian j(q.rows());
  std::cout << std::setprecision(17);
  if (argc == 7) {
    int compliance_segment = -1, sensor_segment = -1;
    for (unsigned i=0; i<chain.segments.size(); ++i) {
      if (chain.segments[i].getName() == argv[5]) compliance_segment = i+1;
      if (chain.segments[i].getName() == argv[6]) sensor_segment = i+1;
    }
    if (compliance_segment < 0 || sensor_segment < 0) return 5;
    Eigen::VectorXd velocity = Eigen::VectorXd::Zero(q.rows());
    Eigen::Matrix<double,6,1> last_error = Eigen::Matrix<double,6,1>::Zero();
    Eigen::Matrix<double,6,1> selection;
    selection << 1.,1.,0.5,1.,0.5,1.;
    KDL::Frame cached_pose;
    bool initialized = false;
    while (std::cin >> q(0)) {
      for (unsigned i=1; i<q.rows(); ++i) std::cin >> q(i);
      double target_values[7];
      for (double& x : target_values) std::cin >> x;
      KDL::Frame target(KDL::Rotation::Quaternion(target_values[3],target_values[4],target_values[5],target_values[6]),
                        KDL::Vector(target_values[0],target_values[1],target_values[2]));
      KDL::Wrench sensor;
      for (int i=0; i<6; ++i) std::cin >> sensor[i];
      Eigen::Matrix<double,6,1> stiffness,p,d;
      for (int i=0; i<6; ++i) std::cin >> stiffness[i];
      for (int i=0; i<6; ++i) std::cin >> p[i];
      for (int i=0; i<6; ++i) std::cin >> d[i];
      double scale;
      int iterations;
      std::cin >> scale >> iterations;
      if (!initialized) { forward.JntToCart(q,cached_pose); initialized = true; }
      for (int iteration=0; iteration<iterations; ++iteration) {
        KDL::Vector delta = target.p-cached_pose.p, axis;
        double distance = delta.Normalize();
        double angle = (target.M*cached_pose.M.Inverse()).GetRotAngle(axis);
        delta = delta*std::clamp(distance,-1.,1.);
        axis = axis*std::clamp(angle,-1.,1.);
        Eigen::Matrix<double,6,1> motion;
        for (int i=0;i<3;++i) { motion[i]=delta[i]; motion[i+3]=axis[i]; }
        KDL::Frame compliance,sensor_frame;
        forward.JntToCart(q,compliance,compliance_segment);
        forward.JntToCart(q,sensor_frame,sensor_segment);
        Eigen::Matrix3d rotation;
        for (int row=0;row<3;++row) for(int col=0;col<3;++col) rotation(row,col)=compliance.M(row,col);
        Eigen::Matrix<double,6,1> spring;
        spring.head<3>() = rotation*stiffness.head<3>().asDiagonal()*rotation.transpose()*motion.head<3>();
        spring.tail<3>() = rotation*stiffness.tail<3>().asDiagonal()*rotation.transpose()*motion.tail<3>();
        KDL::Wrench force = compliance.M*((compliance.Inverse()*sensor_frame)*sensor);
        Eigen::Matrix<double,6,1> error;
        for(int i=0;i<6;++i) error[i]=selection[i]*spring[i]+(1-selection[i])*force[i];
        Eigen::Matrix<double,6,1> input=scale*(p.cwiseProduct(error)+d.cwiseProduct((error-last_error)/0.02));
        dynamics.JntToMass(q,h);
        jacobian.JntToJac(q,j);
        Eigen::VectorXd acceleration=h.data.inverse()*j.data.transpose()*input;
        q.data += velocity*0.02;
        velocity = 0.9*(velocity+acceleration*0.02);
        last_error=error;
        forward.JntToCart(q,cached_pose);
      }
      for(unsigned i=0;i<q.rows();++i) std::cout << q(i) << ' ';
      for(unsigned i=0;i<q.rows();++i) std::cout << velocity[i] << ' ';
      for(int i=0;i<6;++i) std::cout << last_error[i] << ' ';
      std::cout << '\n';
    }
    return 0;
  }
  while (std::cin >> q(0)) {
    for (unsigned i=1; i<q.rows(); ++i) std::cin >> q(i);
    dynamics.JntToMass(q, h);
    jacobian.JntToJac(q, j);
    KDL::Frame pose;
    forward.JntToCart(q, pose);
    for (unsigned row=0; row<q.rows(); ++row)
      for (unsigned col=0; col<q.rows(); ++col) std::cout << h(row,col) << ' ';
    for (unsigned row=0; row<6; ++row)
      for (unsigned col=0; col<q.rows(); ++col) std::cout << j(row,col) << ' ';
    for (int row=0; row<3; ++row) std::cout << pose.p[row] << ' ';
    double x,y,z,w;
    pose.M.GetQuaternion(x,y,z,w);
    std::cout << x << ' ' << y << ' ' << z << ' ' << w << '\n';
  }
}
