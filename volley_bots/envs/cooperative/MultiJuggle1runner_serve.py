import os
import time
## OmniDrones part
import hydra
from omegaconf import OmegaConf, DictConfig
import numpy as np
import torch # Version: 2.0.1
from omni_drones.learning import MAPPOPolicy
from tensordict import TensorDict
from torchrl.data import UnboundedContinuousTensorSpec, BoundedTensorSpec
## ROS part
import rospy
from enum import Enum ## autopilot_state_machine
from std_msgs.msg import Empty
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from quadrotor_msgs.msg import ControlCommand, AutopilotFeedback
from quadrotor_msgs.msg import Trajectory, TrajectoryPoint
from quadrotor_msgs.srv import SetElectricFence  ## autoSetFence
## Util part
import copy
import transforms3d
logFile = True
csvFile = True
uavBallXYAlign = True
autoSetFence, shuangQing = True, True


codeDebug = False
# codeDebug = True


timeEncoding = False
resInObs, restitution = False, 0.80
racketCompensate, dist, racBodyPosMocap, racBodyPosOdom = True, np.array([0,0,0.055]), np.array([-0.033,0.025,0.091]), np.array([0.0,0.0,0.055])
brFromImu = True
uavKF = True
ballPosInObs = True
throw = False
humanData = False
OBS_X_OFFSET = 4.5
if humanData:
    codeDebug, useUavKf = True, True
if codeDebug:
    uavBallXYAlign = False
    autoSetFence, shuangQing = False, False
if throw:
    uavBallXYAlign = False

class enum_autopilot_states(Enum):
    OFF=0
    START=1
    HOVER=2
    LAND=3
    EMERGENCY_LAND=4
    BREAKING=5
    GO_TO_POSE=6
    VELOCITY_CONTROL=7
    REFERENCE_CONTROL=8
    TRAJECTORY_CONTROL=9
    COMMAND_FEEDTHROUGH=10
    RC_MANUAL=11

class OmniPolicyRosRunner():
    def __init__(self, cfg):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if not humanData:
            ## make env
            # self.obs_dim = 29 # + 1 # ball mass 
            self.obs_dim = 15 + 3 + 3 + 3 + 3 + 3 + 2 + 2
            if timeEncoding:
                self.obs_dim += 4
            if resInObs:
                self.obs_dim += 1
            observation_spec = UnboundedContinuousTensorSpec(
                shape=torch.Size([1, 1, self.obs_dim]),
                dtype=torch.float32,
                device=self.device)
            action_spec = BoundedTensorSpec(
                low=-1,
                high=1,
                shape=torch.Size([1, 1, 4]),
                dtype=torch.float32,
                device=self.device)
            reward_spec = UnboundedContinuousTensorSpec(
                shape=torch.Size([1, 1, 1]),
                dtype=torch.float32,
                device=self.device)
            agent_spec = AgentSpec(observation_spec, action_spec, reward_spec)
            ## load policy: asymmetric actor critic , comment critic.load in mappo.py
            algos = {"mappo": MAPPOPolicy}
            self.policy = algos[cfg.algo.name.lower()](cfg.algo, agent_spec=agent_spec, device=self.device)
            Server_checkpoint_path = os.path.join(os.path.dirname(__file__), cfg.Server_policy_checkpoint_path)
            self.policy.load_state_dict(torch.load(Server_checkpoint_path, map_location=self.device))
            print(f"Load policy from {Server_checkpoint_path}")
            self.policy.eval()
        ## calc episode
        self.count, self.episode_length, self.ready = 0, 50000000000000000000000000000000000000, False
        ## start ros node
        rospy.init_node('omni2real')
        from mocap_ekf.msg import SimpleOdom
        self.ros_odom, self.last_odom_receive_time, self.odom_timeout = None, None, 1.0/15.0
        # self.odom_sub = rospy.Subscriber("/air/autopilot/state_estimate", Odometry, self.odom_callback)
        self.odom_sub = rospy.Subscriber("/air/autopilot/simple_odom", SimpleOdom, self.odom_callback)
        self.mocap_sub = rospy.Subscriber("/air/vrpn_client_node/air1/pose", PoseStamped, self.mocap_callback)
        self.mocap_sub2 = rospy.Subscriber("/air/vrpn_client_node/air2/pose", PoseStamped, self.mocap_callback2)
        self.ros_mocap, self.ros_mocap2 = None, None
        self.last_mocap_receive_time, self.last_mocap2_receive_time = None, None
        self.mocap_timeout, self.z_offset = 1.0/15.0, None
        self.ball_sub = rospy.Subscriber("/kf0", PoseStamped, self.ball_callback)
        self.ros_ball, self.last_ball_receive_time, self.ball_timeout = None, None, 1.0/15.0
        self.ready_sub = rospy.Subscriber("/ready", Empty, self.ready_callback)
        self.command_pub = rospy.Publisher('/air/autopilot/control_command_input', ControlCommand, queue_size = 10)
        self.force_hover_pub = rospy.Publisher('/air/autopilot/force_hover', Empty, queue_size=10)
        self.force_hover_flag = False
        self.control_freq = 50.0
        self.dt = 1.0 / self.control_freq
        self.warmupFinish = False
        self.autopilot_state = enum_autopilot_states(0)
        self.autopilot_state_sub = rospy.Subscriber("/air/autopilot/feedback", AutopilotFeedback, self.autopilot_feedback_callback,queue_size = 2)
        if uavBallXYAlign:
            self.alignFinish = False
            self.pose_control_pub = rospy.Publisher('/air/autopilot/pose_command', PoseStamped, queue_size=1) 
        else:
            self.alignFinish = True
        ## log + csv
        uuid_str = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime())
        if humanData:
            uuid_str += '_human'
        if csvFile:
            import csv
            csvName ='%s.csv' % uuid_str
            print("csving into", csvName)
            self.csvObj = open(csvName, 'w', newline='')
            self.mycsv = csv.writer(self.csvObj)
            if humanData:
                row = (['uavPos']*3 + ['uavVel']*3  + ['uavMat']*9 + ['rpos']*3 + ['balPos']*3 + ['balVel']*3 + ['time'])
            elif timeEncoding:
                row = (['balPos']*3 + ['uavBr']*3 + ['uavEuler']*3 + 
                        ['obs-uavPos']*3 + ['obs-uavVel']*3 + ['obs-uavWoRa']*3 + ['obs-uavMat']*9 + ['obs-rpos']*3 + ['obs-balVel']*3 + ['obs-time']*4 + 
                        ['rawAct']*4 + ['tarRate']*3 + ['tarThrust'])
            elif resInObs:
                row = (['balPos']*3 + ['uavBr']*3 + ['uavEuler']*3 + 
                        ['obs-uavPos']*3 + ['obs-uavVel']*3 + ['obs-uavWoRa']*3 + ['obs-uavMat']*9 + ['obs-rpos']*3 + ['obs-balVel']*3 + ['obs-res'] +
                        ['rawAct']*4 + ['tarRate']*3 + ['tarThrust'])
            elif ballPosInObs:
                row = (['balPos']*3 + ['uavBr']*3 + ['uavEuler']*3 + 
                        ['obs-uavPos']*3 + ['obs-uavVel']*3 + ['obs-uavMat']*9 + ['obs-balPos']*3 + ['obs-rpos-anchor']*3 + ['obs-rpos-drone']*3 + ['obs-rpos-ball']*3 + ['obs-balVel']*3 + ['turn']*2 + ['id']*2 +
                        ['rawAct']*4 + ['tarRate']*3 + ['tarThrust'] + ['tarAnchor']*3 )
            else:
                row = (['balPos']*3 + ['uavBr']*3 + ['uavEuler']*3 + 
                        ['obs-uavPos']*3 + ['obs-uavVel']*3 + ['obs-uavMat']*9 + ['obs-balPos']*3 + ['obs-rpos-anchor']*3 + ['obs-rpos-drone']*3 + ['obs-rpos-ball']*3 + ['obs-balVel']*3 + ['turn']*2 + ['id']*2 +
                        ['rawAct']*4 + ['tarRate']*3 + ['tarThrust'] + ['tarAnchor']*3)
            self.mycsv.writerow(row)
            self.csvFileOpen = True
        if logFile:
            logName ='%s.log' % uuid_str
            print("logging into", logName)
            self.mylog = open(logName, mode = 'a',encoding='utf-8')
            self.logFileOpen = True
            if not humanData:
                print(f"Load policy from {Server_checkpoint_path}", file=self.mylog)
            else:
                print(f"Collecting human data", file=self.mylog)
        if autoSetFence:
            self.autofenceOK = False
        else:
            self.autofenceOK = True
        self.lastMat = None ## check orient sudden change
        self.freefall = False
        self.aboveCnt, self.stopErrLog = 0, False
        if brFromImu:
            self.imu_sub = rospy.Subscriber("/air/mavros/imu/data", Imu, self.imu_callback)
            self.ros_imu, self.last_imu_receive_time, self.imu_timeout = None, None, 1.0/20.0
        if uavKF:
            self.uav_kf_sub = rospy.Subscriber("/uav_kf", PoseStamped, self.uav_kf_callback)
            self.ros_uav_kf, self.last_uav_kf_receive_time, self.uav_kf_timeout = None, None, 1.0/20.0
        if humanData:
            self.pub_timer = rospy.Timer(rospy.Duration(self.dt), self.human_data_collect)
        else:
            self.pub_timer = rospy.Timer(rospy.Duration(self.dt), self.eval_ros_step)

        # self.ball_anchor = [1, 1.4, 0]
        self.obs_position_offset = np.array([OBS_X_OFFSET, 0.0, 0.0], dtype=np.float32)
        self.ball_anchor = np.array([4.5, -1.25, 1.0], dtype=np.float32)
        self.drone_anchor = np.array([4.5, 1.25, 1.0], dtype=np.float32)
        self.turn = np.array([0.0, 1.0], dtype=np.float32)
        self.last_ball_vel = None
        self.prev_obs_ball_vel = np.zeros(3, dtype=np.float32)
        self.Server_hit = 0
        self.id = np.array([0.0, 1.0], dtype=np.float32)

        

            
    def autopilot_feedback_callback(self, data):
        self.autopilot_state = enum_autopilot_states(data.autopilot_state) # data.autopilot_state -> uint8

    def uav_kf_callback(self, data):
        self.ros_uav_kf, self.last_uav_kf_receive_time = data, data.header.stamp

    def imu_callback(self, data):
        self.ros_imu, self.last_imu_receive_time = data, data.header.stamp

    def odom_callback(self, data):
        self.ros_odom, self.last_odom_receive_time = data, data.header.stamp

    def mocap_callback(self, data):
        self.ros_mocap, self.last_mocap_receive_time = data, data.header.stamp
        
    def mocap_callback2(self, data):
        self.ros_mocap2, self.last_mocap2_receive_time = data, data.header.stamp
    
    def ball_callback(self, data):
        self.ros_ball, self.last_ball_receive_time = data, data.header.stamp
        # ballVz = np.around(self.ros_ball.pose.orientation.z, 4)
        # self.lastBallVz = ballVz

    def ready_callback(self, data):
        ## check mocap & odom valid
        odom_valid = self.ros_odom is not None and self.last_odom_receive_time is not None and (rospy.get_time() - self.last_odom_receive_time.to_sec()) < self.odom_timeout
        mocap_valid = self.ros_mocap is not None and self.last_mocap_receive_time is not None and (rospy.get_time() - self.last_mocap_receive_time.to_sec()) < self.mocap_timeout
        mocap_valid2 = self.ros_mocap2 is not None and self.last_mocap2_receive_time is not None and (rospy.get_time() - self.last_mocap2_receive_time.to_sec()) < self.mocap_timeout
        if not (mocap_valid and odom_valid and mocap_valid2):
            print("mocap or odom invalid, msg ignored!")
            return
        else:
            self.ready = True
            origin = np.array([self.ros_mocap.pose.position.x, 
                                self.ros_mocap.pose.position.y, 
                                self.ros_mocap.pose.position.z])
            self.z_offset = self.ros_mocap.pose.position.z - self.ros_odom.pose.position.z
            readyLogStr = "start now! mocapPos (%.5f, %.5f, %.5f), z offset:%.5f\n" % (origin[0], origin[1], origin[2], self.z_offset)
            print(readyLogStr)
            if self.logFileOpen:
                print(readyLogStr, file=self.mylog)
            self.count = 0
        ## set fence 
        if autoSetFence:
            if shuangQing:  ## for ShuangQing (middle is origin)
                zLow, zHigh = 0.5, 2.5
                # xLow, xHigh = -1.3, 1.3
                xLow, xHigh = -1.0, 2.5
                yLow, yHigh = -0.8, 0.8
                vMax, wMax, dqMax = 6, 10, 8
            x, y, z = origin[0], origin[1], origin[2]
            z_min, z_max = zLow - z, zHigh - z
            x_min, x_max = xLow - x, xHigh - x
            y_min, y_max = yLow - y, yHigh - y
            rospy.wait_for_service('air/set_electric_fence')
            ## wait for hover to set fence
            while not self.autopilot_state==enum_autopilot_states.HOVER:
                time.sleep(0.2)
                print("waiting for hover to set fence", self.autopilot_state)
            fenceLogStr = ""
            try:
                set_fence = rospy.ServiceProxy('air/set_electric_fence', SetElectricFence)
                request = set_fence(1,z_min,z_max,x_min,x_max,y_min,y_max,vMax,wMax,dqMax) ## first 1 for true
                if request.res:
                    self.autofenceOK = True
                    fenceLogStr += "---------------------------------------------\nautoSetFence Success" + str(request)
                    fenceLogStr += 'mocap x ' + str(x) + ' y ' + str(y) + ' z ' + str(z)
                    fenceLogStr += '\nz_min, z_max ' + str(z_min) + ' ' + str(z_max)
                    fenceLogStr += '\nx_min, x_max ' + str(x_min) + ' ' + str(x_max)
                    fenceLogStr += '\ny_min, y_max ' + str(y_min) + ' ' + str(y_max)
                else:
                    self.autofenceOK = False
                    fenceLogStr += "---------------------------------------------\nautoSetFence Fail, return" + str(request)
            except rospy.ServiceException as e:
                self.autofenceOK = False
                fenceLogStr += "---------------------------------------------setFenceService call failed: %s, return" % e
            print(fenceLogStr)
            if self.logFileOpen:
                print(fenceLogStr, file=self.mylog)
            if not self.autofenceOK:
                return
        ## align uav and ball pos
        if uavBallXYAlign:
            beforeAlignLogStr = "---------------------------------------------\n"
            while self.ros_ball is None:
                time.sleep(0.1)
                print("waiting for ball msg to align")
            baVx, baVy, baVz = self.ros_ball.pose.orientation.x, self.ros_ball.pose.orientation.y, self.ros_ball.pose.orientation.z
            while ( abs(baVx) > 0.3 or 
                    abs(baVy) > 0.3 or
                    abs(baVz) > 0.3): 
                time.sleep(0.1)
                print("ball moving! cannot align", np.around(baVx,2), np.around(baVy,2), np.around(baVz,2))
            baPx, baPy, baPz = self.ros_ball.pose.position.x, self.ros_ball.pose.position.y, self.ros_ball.pose.position.z
            uaPxo, uaPyo, uaPzo = self.ros_odom.pose.position.x, self.ros_odom.pose.position.y, self.ros_odom.pose.position.z
            uaPxm, uaPym, uaPzm = self.ros_mocap.pose.position.x, self.ros_mocap.pose.position.y, self.ros_mocap.pose.position.z
            uaPxo_des = self.ros_ball.pose.position.x
            uaPyo_des = self.ros_ball.pose.position.y
            uaPzo_des = 1.0 - self.z_offset
            beforeAlignLogStr += "current ball pos: " + str(np.around(baPx,2)) + ' ' + str(np.around(baPy,2)) + ' ' + str(np.around(baPz,2)) + '\n'
            beforeAlignLogStr += "current uav pos odom: " + str(np.around(uaPxo,2)) + ' ' + str(np.around(uaPyo,2)) + ' ' + str(np.around(uaPzo,2)) + '\n'
            beforeAlignLogStr += "current uav pos mocap: " + str(np.around(uaPxm,2)) + ' ' + str(np.around(uaPym,2)) + ' ' + str(np.around(uaPzm,2)) + '\n'
            beforeAlignLogStr += "target uav pos odom: " + str(np.around(uaPxo_des,2)) + ' ' + str(np.around(uaPyo_des,2)) + ' ' + str(np.around(uaPzo_des,2))
            print(beforeAlignLogStr)
            if self.logFileOpen:
                print(beforeAlignLogStr, file=self.mylog)
            ## prepare msg
            pose_command = PoseStamped()
            # pose_command.pose.position.x = uaPxo_des
            pose_command.pose.position.x = uaPxo_des
            pose_command.pose.position.y = uaPyo_des
            pose_command.pose.position.z = uaPzo_des
            # pose_command.pose.orientation = self.ros_odom.pose.orientation
            pose_command.pose.orientation.w = 1
            pose_command.pose.orientation.x = 0
            pose_command.pose.orientation.y = 0
            pose_command.pose.orientation.z = 0
            ## pub
            while not self.autopilot_state==enum_autopilot_states.HOVER:
                time.sleep(0.1)
                print("waiting for hover to pub pose command", self.autopilot_state)
            first_pub_time = rospy.get_time()
            self.pub_timeout = 0.5
            self.pose_tolerance = 0.02
            pose_gap = np.sqrt( (self.ros_odom.pose.position.x-uaPxo_des)**2 + 
                                (self.ros_odom.pose.position.y-uaPyo_des)**2 + 
                                (self.ros_odom.pose.position.z-uaPzo_des)**2 )
            while (rospy.get_time() - first_pub_time < self.pub_timeout and
                    self.autopilot_state == enum_autopilot_states.HOVER and
                    pose_gap>self.pose_tolerance):
                time.sleep(0.1)
                self.pose_control_pub.publish(pose_command)
                pose_gap = np.sqrt( (self.ros_odom.pose.position.x-uaPxo_des)**2 + 
                                    (self.ros_odom.pose.position.y-uaPyo_des)**2 + 
                                    (self.ros_odom.pose.position.z-uaPzo_des)**2 )
                print("pub pose command")
            ## result
            while not self.autopilot_state==enum_autopilot_states.HOVER:
                time.sleep(0.5)
                print("waiting for hover", self.autopilot_state)
            baPx, baPy, baPz = self.ros_ball.pose.position.x, self.ros_ball.pose.position.y, self.ros_ball.pose.position.z
            uaPxo, uaPyo, uaPzo = self.ros_odom.pose.position.x, self.ros_odom.pose.position.y, self.ros_odom.pose.position.z
            uaPxm, uaPym, uaPzm = self.ros_mocap.pose.position.x, self.ros_mocap.pose.position.y, self.ros_mocap.pose.position.z
            
            afterAlignLogStr = "recall target uav pos odom:"  + ' ' + str(np.around(uaPxo_des,2)) + ' ' + str(np.around(uaPyo_des,2)) + ' ' + str(np.around(uaPzo_des,2)) + '\n'
            afterAlignLogStr += "after align ball pos:" + ' ' + str(np.around(baPx,2)) + ' ' + str(np.around(baPy,2)) + ' ' + str(np.around(baPz,2)) + '\n'
            afterAlignLogStr += "after align uav pos odom:"  + ' ' + str(np.around(uaPxo,2)) + ' ' + str(np.around(uaPyo,2)) + ' ' + str(np.around(uaPzo,2)) + '\n'
            afterAlignLogStr += "after align uav pos mocap:"  + ' ' + str(np.around(uaPxm,2)) + ' ' + str(np.around(uaPym,2)) + ' ' + str(np.around(uaPzm,2)) + '\n'
            afterAlignLogStr += "---------------------------------------------"
            self.alignFinish = True
            print(afterAlignLogStr)
            if self.logFileOpen:
                print(afterAlignLogStr, file=self.mylog)
                 
    def eval_ros_step(self, event = None): ## event paired with pub_timer
        funcInTime = time.time()
        
        ## dealbreaker
        if throw:
            speedLimit = 1
        else:
            speedLimit = -0.5
        if not self.freefall:
            if (self.ros_ball and 
                self.ros_ball.pose.orientation.z < speedLimit and 
                self.ros_mocap is not None and
                np.abs(self.ros_mocap.pose.position.x - self.ros_ball.pose.position.x) < 1 and
                np.abs(self.ros_mocap.pose.position.y - self.ros_ball.pose.position.y) < 1 and
                self.ros_ball.pose.position.z < 2.8 and
                self.ros_ball.pose.position.z > 1.5):
                self.freefall = True
        if not (self.autofenceOK and ## 围栏
                self.ready and ## 指令
                self.count < self.episode_length and ## 到长度
                self.ros_odom and ## 有过odom
                (rospy.get_time() - self.last_odom_receive_time.to_sec()) < self.odom_timeout and ## odom实时
                self.ros_mocap and ## 有过mocap
                self.last_mocap_receive_time is not None and
                (rospy.get_time() - self.last_mocap_receive_time.to_sec()) < self.mocap_timeout and ## air1 mocap实时
                self.ros_mocap2 and ## 有过air2 mocap
                self.last_mocap2_receive_time is not None and
                (rospy.get_time() - self.last_mocap2_receive_time.to_sec()) < self.mocap_timeout and ## air2 mocap实时
                self.ros_ball and ## 有过球速
                (rospy.get_time() - self.last_ball_receive_time.to_sec()) < self.ball_timeout and ## 球速实时
                # self.ros_odom.pose.position.z < self.ros_ball.pose.position.z and ## 机比球低
                self.ros_ball.pose.position.z > 0.1 and ## 球不能太低
                self.freefall  ## 球下落
                # not self.Server_hit
                ):
            if self.stopErrLog or (not self.warmupFinish) or (uavBallXYAlign and not self.alignFinish):
                return
            errLogStr = "ErrLog: "
            if not self.autofenceOK:
                errLogStr += "noFence "
            if not self.ready:
                errLogStr += "noReady "
            if self.count >= self.episode_length:
                errLogStr += "cntOver, no more errLog "
                self.stopErrLog = True
                if self.logFileOpen:
                    print(errLogStr,  file=self.mylog)
                    self.mylog.close()
                    self.logFileOpen = False
                if self.csvFileOpen:
                    self.csvFileOpen = False
                    self.csvObj.close()
            if not self.ros_odom:
                errLogStr += "noOdom "
            elif (rospy.get_time() - self.last_odom_receive_time.to_sec()) >= self.odom_timeout:
                errLogStr += "odomGap %.2f s " % (rospy.get_time() - self.last_odom_receive_time.to_sec())
            if not self.ros_mocap:
                errLogStr += "noMocap "
            elif self.last_mocap_receive_time is None:
                errLogStr += "noMocapStamp "
            elif (rospy.get_time() - self.last_mocap_receive_time.to_sec()) > self.mocap_timeout:
                errLogStr += "mocapGap %.2f s " % (rospy.get_time() - self.last_mocap_receive_time.to_sec())
            if not self.ros_mocap2:
                errLogStr += "noMocap2 "
            elif self.last_mocap2_receive_time is None:
                errLogStr += "noMocap2Stamp "
            elif (rospy.get_time() - self.last_mocap2_receive_time.to_sec()) > self.mocap_timeout:
                errLogStr += "mocap2Gap %.2f s " % (rospy.get_time() - self.last_mocap2_receive_time.to_sec())
            if not self.ros_ball:
                errLogStr += "noBall "
            elif (rospy.get_time() - self.last_ball_receive_time.to_sec()) >= self.ball_timeout:
                errLogStr += "ballGap %.2f s " % (rospy.get_time() - self.last_ball_receive_time.to_sec())
            # if self.ros_odom and self.ros_ball and (self.ros_odom.pose.position.z > self.ros_ball.pose.position.z):
            #     errLogStr += "ballLower %.2f < %.2f " % (self.ros_ball.pose.position.z, self.ros_odom.pose.position.z)
            #     self.aboveCnt += 1
            if self.ros_ball and self.ros_ball.pose.position.z <= 0.1:
                errLogStr += "ballLow %.2f < 0.1 " % self.ros_ball.pose.position.z
            if not self.freefall:
                errLogStr += "noFreefall "
                if self.ros_ball:
                    errLogStr += "%.2f %.2f " % (self.ros_ball.pose.position.z, self.ros_ball.pose.orientation.z)
            print(errLogStr)
            if self.logFileOpen:
                print(errLogStr, file=self.mylog)
            if self.aboveCnt > 100:
                self.stopErrLog = True
                print("aboveCnt %d, no more errLog" % self.aboveCnt)
                if self.logFileOpen:
                    print("aboveCnt %d, no more errLog" % self.aboveCnt, file=self.mylog)
                    self.mylog.close()
                    self.logFileOpen = False
                if self.csvFileOpen:
                    self.csvFileOpen = False
                    self.csvObj.close()
            return
        ## preprocess
        self.count += 1
        runLogStr = ""
        ## warning
        timegap = rospy.get_time() - self.last_mocap_receive_time.to_sec()
        timegap2 = rospy.get_time() - self.last_mocap2_receive_time.to_sec()
        mocap_valid = self.ros_mocap is not None and timegap < self.mocap_timeout
        mocap_valid2 = self.ros_mocap2 is not None and timegap2 < self.mocap_timeout
        useMocap = mocap_valid and mocap_valid2
        ## make obs (unit? frame?)
        odom = copy.deepcopy(self.ros_odom)
        if useMocap and mocap_valid2:
            mocap = copy.deepcopy(self.ros_mocap)
            mocap2 = copy.deepcopy(self.ros_mocap2)
            current_pos = np.array([mocap.pose.position.x, mocap.pose.position.y, mocap.pose.position.z])
            current_pos2 = np.array([mocap2.pose.position.x, mocap2.pose.position.y, mocap2.pose.position.z])
            quaternion = np.array([mocap.pose.orientation.w,
                                mocap.pose.orientation.x,
                                mocap.pose.orientation.y,
                                mocap.pose.orientation.z])
            racketCenter = current_pos + np.array(transforms3d.quaternions.rotate_vector(racBodyPosMocap, quaternion, is_normalized=True))
            if racketCompensate:
                current_pos += np.array(transforms3d.quaternions.rotate_vector(racBodyPosMocap - dist, quaternion, is_normalized=True))
        else:
            if not racketCompensate:
                current_pos = np.array([odom.pose.position.x, odom.pose.position.y, odom.pose.position.z + self.z_offset])
            else:
                current_pos = np.array([odom.pose.position.x, odom.pose.position.y, odom.pose.position.z])
            quaternion = np.array([odom.pose.orientation.w,
                                odom.pose.orientation.x,
                                odom.pose.orientation.y,
                                odom.pose.orientation.z])
            racketCenter = current_pos + np.array(transforms3d.quaternions.rotate_vector(racBodyPosOdom, quaternion, is_normalized=True))         
        euler = np.array(transforms3d.euler.quat2euler(quaternion, axes='rzyx')) * 57.3 ## r连体系
        mat = transforms3d.quaternions.quat2mat(quaternion)
        ## check orient sudden change
        if self.lastMat is None:
            self.lastMat = mat
        else:
            theta = np.arccos((np.trace(mat @ self.lastMat.T)-1)/2)
            if theta > 0.8: ## rad
                mat = self.lastMat
                runLogStr += "orientSuddenChange %.2f deg " % (theta*57.3)
            else:
                self.lastMat = mat
        ball = copy.deepcopy(self.ros_ball)
        ball_pos = np.array([ball.pose.position.x, ball.pose.position.y, ball.pose.position.z])
        ball_vel = np.array([ball.pose.orientation.x, ball.pose.orientation.y, ball.pose.orientation.z])

        ## chk uav_kf
        useUavKf = (uavKF and
                    self.ros_uav_kf is not None and
                    (rospy.get_time() - self.last_uav_kf_receive_time.to_sec()) < self.uav_kf_timeout)
        if useUavKf:
            uav_kf = copy.deepcopy(self.ros_uav_kf)
            uav_vel = np.array([uav_kf.pose.orientation.x, 
                                uav_kf.pose.orientation.y,
                                uav_kf.pose.orientation.z])
        else:
            uav_vel = np.array([odom.twist.linear.x, odom.twist.linear.y, odom.twist.linear.z])

        runLogStr += 'cnt %3d baPo, uaPo ' % self.count
        runLogStr += str(np.around(ball_pos, 2))
        runLogStr += str(np.around(current_pos, 2))
        runLogStr += ' baVe, uaVe '
        runLogStr += str(np.around(ball_vel, 2))
        runLogStr += str(np.around(uav_vel, 2))
        runLogStr += " uaEu "
        runLogStr += str(np.around(euler,1))
        runLogStr += " raCe "
        runLogStr += str(np.around(racketCenter,2))
        runLogStr += " usMo "
        runLogStr += str(useMocap)
        
        if (brFromImu and
            self.ros_imu and
            (rospy.get_time() - self.last_imu_receive_time.to_sec()) < self.imu_timeout):
            imu = copy.deepcopy(self.ros_imu)
            useImu = True
            bodyrate = np.array([imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z])
        else:
            useImu = False
            bodyrate = np.array([odom.twist.angular.x, odom.twist.angular.y, odom.twist.angular.z])
        runLogStr += " usIm "
        runLogStr += str(useImu)
        runLogStr += " usKf "
        runLogStr += str(useUavKf)
        runLogStr += " "

        rpos = ball_pos - current_pos
        ball_rpos = ball_pos - self.ball_anchor 
        ball_mass = np.array([0.03])
        #check hit
        ball_near_drone_threshold = 0.2      

        if self.last_ball_vel is not None:
            print("get last ball vel")
            delta_vz = (ball_vel[..., 2] - self.last_ball_vel[..., 2] > 9.8 * 0.02)
            delta_vxy = np.linalg.norm(ball_vel[..., :2] - self.last_ball_vel[..., :2], axis=-1) > 0.3
            distance = np.linalg.norm(rpos)
            if delta_vz and delta_vxy and distance < ball_near_drone_threshold:
                self.Server_hit = 1
                if self.logFileOpen:
                    print("Drone hit ball", file=self.mylog)
            if delta_vz :
                print("delta_vz meet requirement")
                if self.logFileOpen:
                    print("delta_vz meet requirement", file=self.mylog)
            if delta_vxy :
                print("delta_vxy meet requirement")
                if self.logFileOpen:
                    print("delta_vxy meet requirement", file=self.mylog)
            if distance < ball_near_drone_threshold :
                print("distance < ball_near_drone_threshold meet requirement")
                if self.logFileOpen:
                    print("distance < ball_near_drone_threshold meet requirement", file=self.mylog)
                if not self.force_hover_flag:
                    self.force_hover_flag = True
                    self.force_hover_pub.publish(Empty()) # force hover
                    print("force hover! return")

                # return

        self.last_ball_vel = ball_vel
        print("[DEBUG] self.last_ball_vel :", self.last_ball_vel)


        obs_array = np.zeros((1,1,self.obs_dim), dtype = np.float32)

        obs_current_pos = current_pos + self.obs_position_offset
        obs_ball_pos = ball_pos + self.obs_position_offset
        rpos_anchor = obs_current_pos - self.drone_anchor
        rpos_drone = current_pos2 - current_pos
        rpos_ball = obs_current_pos - obs_ball_pos

        obs_parts = [
            obs_current_pos,
            uav_vel,
            mat.flatten(order='F'),
            obs_ball_pos,
            rpos_anchor,
            rpos_drone,
            rpos_ball,
            ball_vel,
            self.turn,
            self.id,
        ]
        if timeEncoding:
            time_encoding = np.ones(4) * self.count / self.episode_length
            obs_parts.append(time_encoding)
        if resInObs:
            obs_parts.append(np.array([restitution], dtype=np.float32))
        obs_array[0, 0, :] = np.concatenate([part.flatten() for part in obs_parts], axis=-1)
        obs_tensordict = TensorDict(
            {"observation": torch.tensor(obs_array, dtype=torch.float32, device=self.device)}, 
            batch_size=torch.Size([1]), 
            device=self.device)
        input_tensordict = TensorDict(
            {"agents": obs_tensordict}, 
            batch_size=torch.Size([1]),
            device=self.device)
        ## inference
        output_tensordict = self.policy(input_tensordict, deterministic=True)
        action = output_tensordict["agents"]["action"]
        save_raw_action = _t2n(action)
        ## postprocess (move and scale)
        action = torch.tanh(action)
        target_rate, target_thrust = action.split([3, 1], -1)
        target_thrust = (target_thrust + 1) / 2
        target_rate, target_thrust = _t2n(target_rate.squeeze()), _t2n(target_thrust.squeeze())
        target_thrust *= 20
        target_rate *= np.pi
        ## fix ctbr for debug
        # target_rate[0] = target_rate[1] = target_rate[2] = 0.0
        # target_thrust = 9.95
        ## pub cmd
        cmd_msg = ControlCommand()
        cmd_msg.header.stamp = rospy.Time.now()
        cmd_msg.control_mode = cmd_msg.BODY_RATES
        cmd_msg.armed = True
        cmd_msg.expected_execution_time = rospy.Time(0.020)
        cmd_msg.bodyrates.x = target_rate[0]
        # cmd_msg.bodyrates.x = 0.0
        cmd_msg.bodyrates.y = target_rate[1]
        cmd_msg.bodyrates.z = target_rate[2]
        # cmd_msg.bodyrates.z = 0.0
        cmd_msg.collective_thrust = target_thrust
        if not self.Server_hit:
            self.command_pub.publish(cmd_msg)
        ### pub log
        if csvFile and self.csvFileOpen:
            baPo = ball_pos
            uaBr = bodyrate
            uaEu = euler
            obs = obs_array.squeeze()
            raAc = save_raw_action.squeeze()
            taRa = target_rate
            taTh = np.array([target_thrust])
            taAn = self.ball_anchor
            row = np.concatenate((baPo,uaBr,uaEu,obs,raAc,taRa,taTh,taAn), axis=-1)
            self.mycsv.writerow(row.tolist())

        funcOutTime = time.time()
        duration = (funcOutTime - funcInTime) * 1000
        runLogStr += 'taRa '
        runLogStr += str(np.around(target_rate,1))
        runLogStr += ' taTh %.1f ' % (duration)
        runLogStr += 'taAn '
        runLogStr += str(np.around(self.ball_anchor,1))
        runLogStr += 'balrpos '
        runLogStr += str(np.around(ball_rpos,1))
        runLogStr += 'dura %.1f ms' % (duration)
        if self.logFileOpen:
            print(runLogStr, file=self.mylog)
        
        ## last
        durationPlus = (time.time() - funcInTime) * 1000
        runLogStr += ' dura+ %.1f ms' % (durationPlus)
        print(runLogStr)



    def human_data_collect(self, enevt=None):
        funcInTime = time.time()
        ## dealbreaker
        if not self.freefall:
            if (self.ros_ball and 
                self.ros_ball.pose.orientation.z < -0.5 and 
                self.ros_mocap is not None and
                np.abs(self.ros_mocap.pose.position.x - self.ros_ball.pose.position.x) < 0.5 and
                np.abs(self.ros_mocap.pose.position.y - self.ros_ball.pose.position.y) < 0.5):
                self.freefall = True
        if not (self.ros_mocap and ## 有过mocap
                (rospy.get_time() - self.last_mocap_receive_time.to_sec()) < self.mocap_timeout and ## mocap实时
                self.ros_ball and ## 有过球速
                (rospy.get_time() - self.last_ball_receive_time.to_sec()) < self.ball_timeout and ## 球速实时
                 self.ros_ball.pose.position.z > 0.4 and ## 球不能太低
                self.freefall and ## 球下落
                self.ros_uav_kf and ## 有过机速
                (rospy.get_time() - self.last_uav_kf_receive_time.to_sec()) < self.uav_kf_timeout ## 机速实时
                ):
            if self.stopErrLog:
                return
            errLogStr = "ErrLog: "
            if not self.ros_mocap:
                errLogStr += "noMocap "
            elif (rospy.get_time() - self.last_mocap_receive_time.to_sec()) > self.mocap_timeout:
                errLogStr += "mocapGap %.2f s " % (rospy.get_time() - self.last_mocap_receive_time.to_sec())
            if not self.ros_ball:
                errLogStr += "noBall "
            elif (rospy.get_time() - self.last_ball_receive_time.to_sec()) >= self.ball_timeout:
                errLogStr += "ballGap %.2f s " % (rospy.get_time() - self.last_ball_receive_time.to_sec())
            if self.ros_ball and self.ros_ball.pose.position.z <= 0.4:
                errLogStr += "ballLow %.2f < 0.4 " % self.ros_ball.pose.position.z
                self.aboveCnt += 1
            if not self.freefall:
                errLogStr += "noFreefall "
                if self.ros_ball:
                    errLogStr += "%.2f %.2f " % (self.ros_ball.pose.position.z, self.ros_ball.pose.orientation.z)
            if not self.ros_uav_kf:
                errLogStr += "noUavKf "
            elif (rospy.get_time() - self.last_uav_kf_receive_time.to_sec()) > self.uav_kf_timeout:
                errLogStr += "uavKfGap %.2f s " % (rospy.get_time() - self.last_uav_kf_receive_time.to_sec())
            print(errLogStr)
            if self.logFileOpen:
                print(errLogStr, file=self.mylog)
            if self.aboveCnt > 5:
                self.stopErrLog = True
                print("aboveCnt %d, no more errLog" % self.aboveCnt)
                if self.logFileOpen:
                    print("aboveCnt %d, no more errLog" % self.aboveCnt, file=self.mylog)
                    self.mylog.close()
                    self.logFileOpen = False
                if self.csvFileOpen:
                    self.csvFileOpen = False
                    self.csvObj.close()
            return
        ## preprocess
        self.count += 1
        runLogStr = ""
        ## warning
        mocap = copy.deepcopy(self.ros_mocap)
        mocap2 = copy.deepcopy(self.ros_mocap2)
        current_pos = np.array([mocap.pose.position.x, mocap.pose.position.y, mocap.pose.position.z])
        current_pos2 = np.array([mocap2.pose.position.x, mocap2.pose.position.y, mocap2.pose.position.z])
        quaternion = np.array([mocap.pose.orientation.w,
                            mocap.pose.orientation.x,
                            mocap.pose.orientation.y,
                            mocap.pose.orientation.z])
        racketCenter = current_pos + np.array(transforms3d.quaternions.rotate_vector(racBodyPosMocap, quaternion, is_normalized=True))
        if racketCompensate:
            current_pos += np.array(transforms3d.quaternions.rotate_vector(racBodyPosMocap - dist, quaternion, is_normalized=True)) 
        euler = np.array(transforms3d.euler.quat2euler(quaternion, axes='rzyx')) * 57.3 ## r连体系
        mat = transforms3d.quaternions.quat2mat(quaternion)
        ## check orient sudden change
        if self.lastMat is None:
            self.lastMat = mat
        else:
            theta = np.arccos((np.trace(mat @ self.lastMat.T)-1)/2)
            if theta > 0.8: ## rad
                mat = self.lastMat
                runLogStr += "orientSuddenChange %.2f deg " % (theta*57.3)
            else:
                self.lastMat = mat
        ball = copy.deepcopy(self.ros_ball)
        ball_pos = np.array([ball.pose.position.x, ball.pose.position.y, ball.pose.position.z])
        ball_vel = np.array([ball.pose.orientation.x, ball.pose.orientation.y, ball.pose.orientation.z])
        
        ## chk uav_kf
        uav_kf = copy.deepcopy(self.ros_uav_kf)
        uav_vel = np.array([uav_kf.pose.orientation.x, 
                            uav_kf.pose.orientation.y,
                            uav_kf.pose.orientation.z])

        runLogStr += 'cnt %3d baPo, uaPo ' % self.count
        runLogStr += str(np.around(ball_pos, 2))
        runLogStr += str(np.around(current_pos, 2))
        runLogStr += ' baVe, uaVe '
        runLogStr += str(np.around(ball_vel, 2))
        runLogStr += str(np.around(uav_vel, 2))
        runLogStr += " uaEu "
        runLogStr += str(np.around(euler,1))
        runLogStr += " raCe "
        runLogStr += str(np.around(racketCenter,2))

        rpos = ball_pos - current_pos
        collectData = np.concatenate((current_pos.flatten(), 
                                        uav_vel.flatten(), 
                                        mat.flatten(order='F'), 
                                        rpos.flatten(),
                                        ball_pos.flatten(),
                                        ball_vel.flatten(),
                                        np.array([funcInTime])), axis=-1)
        # print(collectData, '\n', mat, '\n', mat.flatten(order='F'))
        
        ### pub log
        if csvFile and self.csvFileOpen:
            self.mycsv.writerow(collectData.tolist())

        funcOutTime = time.time()
        duration = (funcOutTime - funcInTime) * 1000
        runLogStr += ' dura %.1f ms' % (duration)
        if self.logFileOpen:
            print(runLogStr, file=self.mylog)
        
        ## last
        durationPlus = (time.time() - funcInTime) * 1000
        runLogStr += ' dura+ %.1f ms' % (durationPlus)
        print(runLogStr)
            
    def random_inference(self, manual=False): # ros无关；测试policy是否正确导入、可以推理；空转一次、完成cuda的初始化
        if not manual:
            obs = torch.tensor(np.ones((1,1,self.obs_dim)), dtype=torch.float32, device=self.device)
            obs = torch.ones(size=[1, 1, self.obs_dim], dtype=torch.float32, device=self.device)
            print(obs.shape, obs, end=' ')
        else: 
            obs = torch.tensor([[[0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0]]], dtype=torch.float32, device=self.device)
        obs_tensordict = TensorDict(
            {"observation": obs}, 
            batch_size=torch.Size([1]), 
            device=self.device)
        input_tensordict = TensorDict(
            {"agents": obs_tensordict}, 
            batch_size=torch.Size([1]),
            device=self.device)
        timeBeforeInf = time.time()
        output_tensordict = self.policy(input_tensordict, deterministic=True) # inference
        timeAfterInf = time.time()
        action = output_tensordict["agents"]["action"]
        print('random inference time: ', np.around((timeAfterInf-timeBeforeInf) * 1000, 2), 'ms', 'raw action:', _t2n(action))

CONFIG_PATH = os.path.dirname(__file__)
@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="MultiJuggle_serve.yaml")
def main(cfg: DictConfig):
    # print(OmegaConf.to_yaml(cfg))
    runner = OmniPolicyRosRunner(cfg)
    if not humanData:
        print('----------------------------\nwarm up start')
        runner.random_inference(manual=False)
        runner.random_inference(manual=False)
        # runner.random_inference(manual=True)
        # runner.random_inference(manual=True)
        print('warm up finish\n----------------------------')
    runner.warmupFinish = True
    rospy.spin()

def _t2n(x): 
    return x.detach().cpu().numpy()

class AgentSpec:
    def __init__(self, observation_spec, action_spec, reward_spec, n=1, name="drone"):
        self.observation_spec, self.action_spec, self.reward_spec, self.n, self.name = observation_spec, action_spec, reward_spec, n, name

if __name__ == "__main__":
    main()
