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

import rospy
import actionlib
import petri.msg
from geometry_msgs.msg import Point

class SimActionServer:
    def __init__(self, topic, times, interruptible = True):
        self._as = actionlib.SimpleActionServer(topic, petri.msg.SimAction, execute_cb=self.execute_cb, auto_start=False)
        self._interrupt_ok = interruptible
        self._times = times
        self._as.start()

    def execute_cb(self, goal):
        rospy.loginfo("Got request for action {}".format(goal.name))
        feedback = petri.msg.SimFeedback()
        result = petri.msg.SimResult()
        start = rospy.Time.now()
        success = True

        r= rospy.Rate(10)
        while rospy.Time.now()-start<self._times[goal.name] and not rospy.is_shutdown():
            if self._interrupt_ok and self._as.is_preempt_requested():
                self._as.set_preempted()
                success=False
                break
            feedback.time_running=rospy.Time.now()-start
            self._as.publish_feedback(feedback)
            r.sleep()

        if success:
            result.time_running = rospy.Time.now()-start
            self._as.set_succeeded(result)
            rospy.loginfo("Action {} complete".format(goal.name))
        else:
            rospy.loginfo("Action {} preempted or failed after {} seconds".format(goal.name, feedback.time_running.to_sec()))
            
        


class RobotInterface:
    def __init__(self):
        rospy.loginfo("Waiting for movement playback service")
        self._pkl_player = actionlib.SimpleActionClient('/petri/sim/move', petri.msg.SimAction)
        self._pkl_player.wait_for_server()

        rospy.loginfo("Waiting for speech service")
        self._speech_player = actionlib.SimpleActionClient('/petri/sim/say', petri.msg.SimAction)
        self._speech_player.wait_for_server()

        rospy.loginfo("Waiting for lookat service")
        self._lookat_player = actionlib.SimpleActionClient('/petri/sim/look', petri.msg.SimAction)
        self._lookat_player.wait_for_server()

        rospy.loginfo("All services started")
        
        self._face = {"x":0,"y":0,"size":0}
        self._face_sub = rospy.Subscriber('/petri/main_face', Point, self._face_cb)


    def get_lookat_client(self):
        return self._lookat_player
        
    def get_lookat_goal(self, frame, x, y, z):
        goal=petri.msg.SimGoal()
        goal.name = frame
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
        return petri.msg.SimGoal(name=filename)

    def get_speech_client(self):
        return self._speech_player

    def get_speech_goal(self, text):
        return petri.msg.SimGoal(name=text)
        
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
            "wave":"wave",
            "rotate":"rotate"
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
        return name,0,0,0
        
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
        return "hello"

    def get_ok_text(self):
        return "ok"
        
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
    

