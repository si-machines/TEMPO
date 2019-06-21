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
from petri.msg import Detection
from geometry_msgs.msg import Point


class CVImageSubscriber:
    def __init__(self):
        self._pub_on=rospy.Publisher("/petri/faces", Detection, queue_size=1)
        self._pub_off=rospy.Publisher("/petri/no_faces", Detection, queue_size=1)
        self._face_pub=rospy.Publisher("/petri/main_face", Point, queue_size=1)
        
    def loop(self):
        face_x = 1.0
        face_y = 1.0
        biggest_face_size = 1.0
        face_detected = True
        
        if face_detected:
            self._pub_on.publish(Detection(Detection.ON))
            self._pub_off.publish(Detection(Detection.OFF))
            self._face_pub.publish(Point(x=face_x,y=face_y,z=biggest_face_size))
        else:
            self._pub_on.publish(Detection(Detection.OFF))
            self._pub_off.publish(Detection(Detection.ON))
            self._face_pub.publish(Point(x=0,y=0,z=0))

if __name__=="__main__":
    rospy.init_node("background_subtraction_features")
    c = CVImageSubscriber()
    
    while not rospy.is_shutdown():
        c.loop()
        rospy.sleep(0.1)
