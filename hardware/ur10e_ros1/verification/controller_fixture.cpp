#include <cartesian_compliance_controller/cartesian_compliance_controller.h>
#include <hardware_interface/joint_command_interface.h>
#include <geometry_msgs/WrenchStamped.h>
#include <std_srvs/Trigger.h>
#include <fstream>
#include <iomanip>
#include <thread>
#include <chrono>
using HW=hardware_interface::PositionJointInterface;
using Parent=cartesian_compliance_controller::CartesianComplianceController<HW>;
class Fixture: public Parent {
 public:
 using Parent::MotionBase::targetFrameCallback;
 std::vector<double> sensor(){ std::vector<double> x(18); Parent::ForceBase::copyForceState(x,0); return x; }
};
int main(int argc,char**argv){
 const char* master=std::getenv("ROS_MASTER_URI");
 if(!master || std::string(master)!="http://127.0.0.1:11349") return 20;
 if(argc!=2) return 21;
 ros::init(argc,argv,"offline_fdcc_fixture");
 ros::NodeHandle nh("/fdcc"); std::vector<std::string> names;
 if(!nh.getParam("joints",names)||names.size()!=6) return 22;
 std::ifstream in(argv[1]); double q[6],qd[6]={},eff[6]={},cmd[6]={};
 for(int j=0;j<6;j++) if(!(in>>q[j]))return 23;
 HW hw;
 for(int j=0;j<6;j++) hw.registerHandle(hardware_interface::JointHandle(hardware_interface::JointStateHandle(names[j],q+j,qd+j,eff+j),cmd+j));
 Fixture c; if(!c.init(&hw,nh)) return 24;
 c.starting(ros::Time::now());
 ros::AsyncSpinner spinner(1); spinner.start();
 auto pub=nh.advertise<geometry_msgs::WrenchStamped>("ft_sensor_wrench",1);
 auto tare=nh.serviceClient<std_srvs::Trigger>("signal_taring");
 if(!tare.waitForExistence(ros::Duration(3)))return 25;
 std_srvs::Trigger t; if(!tare.call(t)||!t.response.success)return 26;
 for(int i=0;i<300 && pub.getNumSubscribers()==0;i++)std::this_thread::sleep_for(std::chrono::milliseconds(10));
 std::cout<<std::setprecision(17);
 int row=0;
 while(in>>q[0]){
  for(int j=1;j<6;j++)in>>q[j];
  double p[7],w[6];for(auto&v:p)in>>v;for(auto&v:w)in>>v;
  geometry_msgs::PoseStamped target;target.header.frame_id="base_link";target.header.stamp=ros::Time::now();
  target.pose.position.x=p[0];target.pose.position.y=p[1];target.pose.position.z=p[2];
  target.pose.orientation.x=p[3];target.pose.orientation.y=p[4];target.pose.orientation.z=p[5];target.pose.orientation.w=p[6];
  c.targetFrameCallback(target);
  geometry_msgs::WrenchStamped msg;msg.header.frame_id="ati_sensing_frame";msg.header.stamp=ros::Time::now();
  msg.wrench.force.x=w[0];msg.wrench.force.y=w[1];msg.wrench.force.z=w[2];msg.wrench.torque.x=w[3];msg.wrench.torque.y=w[4];msg.wrench.torque.z=w[5];
  pub.publish(msg);
  bool ready=false;
  for(int k=0;k<300;k++){auto got=c.sensor();ready=true;for(int j=0;j<6;j++)ready=ready && got[j]==w[j];if(ready)break;std::this_thread::sleep_for(std::chrono::milliseconds(1));}
  if(!ready)return 27;
  c.update(ros::Time::now(),ros::Duration(.002));
  std::cout<<"ROW "<<row++;for(auto v:cmd)std::cout<<" "<<v;std::cout<<std::endl;
 }
 c.stopping(ros::Time::now()); spinner.stop(); return 0;
}
