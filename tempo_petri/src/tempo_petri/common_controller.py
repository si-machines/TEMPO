#!/usr/bin/env python

# BSD 3-Clause License
#
# Copyright (c) 2019, SIM Lab
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Author: Elaine Short

import rospy
import actionlib
import hlpr_record_demonstration.msg as record_msgs
import hlpr_dialogue_production.msg as dialogue_msgs
import hlpr_lookat.msg as lookat_msgs

from geometry_msgs.msg import Point
from hlpr_lookat.srv import LookAtTS

#from hlpr_manipulation_utils import arm_moveit2 as arm_moveit
from geometry_msgs.msg import TransformStamped
import math
import random

#CAMERA_X_FOV=84.1
CAMERA_X_FOV=74.1
CAMERA_Y_FOV=53.8

XRANGE=2.0
ZDIST = XRANGE/(2*math.tan(math.radians(CAMERA_X_FOV/2)))
YRANGE = 2*ZDIST*math.tan(math.radians(CAMERA_Y_FOV/2))

#print XRANGE
#print YRANGE
#print ZDIST


class RobotInterface:
    def __init__(self):
        rospy.loginfo("Waiting for movement playback service")
        self._pkl_player = actionlib.SimpleActionClient('/playback_keyframe_demo', record_msgs.PlaybackKeyframeDemoAction)
        self._pkl_player.wait_for_server()

        rospy.loginfo("Waiting for speech service")
        self._speech_player = actionlib.SimpleActionClient('/HLPR_Dialogue', dialogue_msgs.DialogueActAction)
        self._speech_player.wait_for_server()

        rospy.loginfo("Waiting for lookat service")
        self._lookat_player = actionlib.SimpleActionClient('/lookat_waypoints_action_server', lookat_msgs.LookatWaypointsAction)
        self._lookat_player.wait_for_server()

        rospy.loginfo("All services started")
        
        self._face = {"x":0,"y":0,"size":0}
        self._face_sub = rospy.Subscriber('/petri/main_face', Point, self._face_cb)


    def get_lookat_client(self):
        return self._lookat_player
        
    def get_lookat_goal(self, frame, x, y, z):
        goal = lookat_msgs.LookatWaypointsGoal()
        target = TransformStamped()
        target.child_frame_id = frame
        target.transform.translation.x = x
        target.transform.translation.y = y
        target.transform.translation.z = z
        goal.scan_positions = [target]
        goal.scan_times = [rospy.Duration(2.0)]
        return goal

    def lookat(self, frame, x, y, z, wait=False):
        self._lookat_player.send_goal(self.get_lookat_goal(frame, x, y ,z))
        if wait:
            self._lookat_player.wait_for_result()
            return self._lookat_player.get_result()
        else:
            return None

    def get_movement_client(self):
        return self._pkl_player

    def get_movement_goal(self, filename):
        return record_msgs.PlaybackKeyframeDemoGoal(bag_file_name=filename)

    def get_speech_client(self):
        return self._speech_player

    def get_speech_goal(self, text):
        return dialogue_msgs.DialogueActGoal(text_or_key=text)
        
    def _face_cb(self, point):
        self._face["x"]=point.x
        self._face["y"]=point.y
        self._face["size"]=point.z
        
    def rotate_object(self, wait = False):
        rospy.loginfo("Rotating object")
        return self.play_pkl(self.get_pkl("rotate"),
                             wait)

    def get_pkl(self, name):
        pkls = {
            "wave":"/home/eshort/robot_movements/wave_joint.pkl",
            "rotate":"/home/eshort/robot_movements/rot_90_deg_joint.pkl",
            }
        return pkls[name]
    
    def wave(self, wait=False):
        rospy.loginfo("Waving")
        return self.play_pkl(self.get_pkl("wave"),
                             wait)

    def play_pkl(self, filename, wait=False):
        #self._pkl_player.cancel_all_goals()
        self._pkl_player.send_goal(self.get_movement_goal(filename))
        if wait:
            self._pkl_player.wait_for_result()
            return self._pkl_player.get_result()
        else:
            return None

    def get_target_pos(self, name):
        locations = {
            "eef":("robotiq_85_left_finger_link",0,0,0),
            "object":("base_link",1.0,0,1.0),
            "forward":("pan_base_link", 1.0, 0.0, 0.05),
            }
                     
        if name=="face":
            if self._face["size"]==0:
                rospy.logwarn("No face detected")
                return None
            else:
                return "kinect_rgb_optical_frame", self._face["x"]*(XRANGE/2), self._face["y"]*(YRANGE/2), ZDIST
        else:
            return locations[name]

        
    def lookat_eef(self, wait=False):
        rospy.loginfo("Looking at EEF")
        frame, x,y,z = self.get_target_pos("eef")
        return self.lookat(frame,x,y,z,wait)
        
    def lookat_object(self, wait=False):
        rospy.loginfo("Looking at object")
        frame, x,y,z = self.get_target_pos("object")
        return self.lookat(frame,x,y,z,wait)

    def look_ahead(self, wait=False):
        rospy.loginfo("Looking ahead")
        frame, x,y,z = self.get_target_pos("forward")
        return self.lookat(frame,x,y,z,wait)
    
    def lookat_face(self, wait=False):
        rospy.loginfo("Looking at face")
        ret = self.get_target_pos("face")
        if ret != None:
            frame, x,y,z = ret
            return self.lookat(frame,x,y,z,wait)
        else:
            return None
        
    def say(self, text, wait = False):
        self._speech_player.send_goal(self.get_speech_goal(text))
        if wait:
            self._speech_player.wait_for_result()
            return self._speech_player.get_result()
        else:
            return None

    def get_hello_text(self):
        greet = ["Hello.", "Hi.", "Hi there.", "Howdy.", "Greetings."]
        explain = ["Welcome to the E.E.R. building.",
                   "My name is Polly.",
                   "I am taking images of this cup.",
                   "I am learning to work near people."]

        return random.choice(greet)+" "+random.choice(explain)

    def get_ok_text(self):
        return "Done."
        
    def say_hello(self, wait=False):
        rospy.loginfo("Saying hello")        
        return self.say(self.get_hello_text(), wait)
        
    def say_done(self, wait=False):
        rospy.loginfo("Saying done")
        return self.say(self.get_ok_text(), wait)

    def wait_for_move_done(self):
        self._pkl_player.wait_for_result()
        return self._pkl_player.get_result()

    def wait_for_speech_done(self):
        self._speech_player.wait_for_result()
        return self._speech_player.get_result()

    def wait_for_lookat_done(self):
        self._lookat_player.wait_for_result()
        return self._lookat_player.get_result()
        
if __name__=="__main__":
    rospy.init_node("robot_interface_test")
    rob = RobotInterface()

    rob.rotate_object()
    #rospy.sleep(2)
    #rob.wave()
