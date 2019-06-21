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
from petri.timed_petri_net import InteractionPetri, PetriNet, PetriSimNode
from petri.msg import Detection
import hlpr_dialogue_production.msg as dialogue_msgs
import actionlib
import numpy as np
import copy
import math
import random
import sys
import networkx as nx
import pygraphviz as pgv
import itertools


class ROSPetri:
    def __init__(self, petri, detector_topics, action_server_info, rate):
        self._ip = petri
        self._r = rospy.Rate(rate)
        self._active_actions = []
        self._action_servers = action_server_info
        self._detector_states = {}
        for detector, topic in detector_topics.items():
            self._detector_states[detector]=False
            rospy.Subscriber(topic, Detection, self.handle_detector, (detector,))
        petri.start()

    def handle_detector(self, msg, args):
        detector = args[0]
        #print detector, msg
        if msg.status == Detection.ON:
            self._ip.detector_on(detector)
            if not self._detector_states[detector]:
                rospy.loginfo("Detector " + detector + " turned on")
                self._detector_states[detector]=True
        else:
            self._ip.detector_off(detector)
            if self._detector_states[detector]:
                rospy.loginfo("Detector " + detector + " turned off")
                self._detector_states[detector]=False
            

    def start_action(self,action):
        rospy.loginfo("Petri net starting action {}".format(action))
        if action in self._action_servers:
            client, goal_cb = self._action_servers[action]
            client.send_goal(goal_cb())
        else:
            rospy.logwarn("No action server provided for action {}".format(action))

    def interrupt_action(self,action):
        rospy.loginfo("Petri net interrupting action {}".format(action))
        if action in self._action_servers:
            client, goal_cb = self._action_servers[action]
            client.cancel_all_goals()
        else:
            rospy.logwarn("No action server provided for action {}".format(action))

    def run(self):
        active = []
        status = [[] for i in range(10)]
        status_msg = ""
        while not rospy.is_shutdown() and not self._ip.stopped():
            #print status[5]
            if status[0]=="interrupt":
                interrupted = filter(lambda p: "timeout" not in p[0] and "interrupt" not in p[0] and "+" not in p[0], status[4])
                if len(interrupted)>0:
                    status_msg += "interrupted: "
                    for action,time in interrupted:
                        self.interrupt_action(action)
                        self._active_actions.remove(action)
                        status_msg+=action+" "
            waiting_actions = self._ip.get_waiting_actions()
            if len(waiting_actions)>0:
                unique_waiting = list(set(map(lambda s: s[1], waiting_actions)))
                for a in unique_waiting:
                    matches = filter(lambda s: s[1]==a, waiting_actions)
                    waiting_for = map(lambda s: s[2], matches)
                    waiting_time = map(lambda s: s[0], matches)

            active = self._ip.get_active_actions()
            
            for time,action in active:
                if action not in self._active_actions:
                    self.start_action(action)
                    self._active_actions.append(action)
            
            for action in self._active_actions:
                if action not in map(lambda s: s[1], active):
                    rospy.loginfo("Petri net action {} completed".format(action))
                    self._active_actions.remove(action)

            
            timeouts = filter(lambda s: "timeout" in s, status[5])
            for t in timeouts:
                info = t.split("+")
                rospy.loginfo("Petri net action {} timed out waiting on {}".format(info[0],info[1]))
            
                    
            status = self._ip.step(1)
            self._r.sleep()
            #if len(status_msg)>0:
            #    rospy.loginfo(status_msg)


    def run_test_no_prune(self, steps, root):
        open_set = set([root])
        seen={}
        reach = nx.DiGraph()

        time = 0
        while len(open_set)>0:
            current = open_set.pop()
            next_nodes = current.get_next()
            if time % 10 == 0:
                print "open/seen:", len(open_set)+1, len(seen)
            for node in next_nodes:
                if node.get_state() in seen:
                    reach.add_edge(current,seen[node.get_state()])
                else:
                    reach.add_node(node)
                    seen[node.get_state()]=node
                    reach.add_edge(current, node)
                    if time < steps:
                        open_set.add(node)
                time+=1
        return reach


    def run_test_prune(self, steps, root):

        open_set0 = {root.key():root}
        open_set1 = {}
        expanded_set = {}
        reach = nx.DiGraph()

        time = 0
        while len(open_set0)+len(open_set1)>0:
            if len(open_set0)==0:
                open_set0=open_set1
                open_set1={}
            current = open_set0.pop(open_set0.keys()[0])
            expanded_set[current.key()]=current
            next_nodes = current.get_next()

            if time % 10 == 0:
                print "current/next/closed:", len(open_set0),len(open_set1), len(expanded_set)

            for node in next_nodes:
                node_key = node.key()
                time+=1
                
                if node_key in expanded_set:
                    #print "Matched node:\n{}\n\n with node: \n{}".format(node, expanded_set[node_key])

                    exact = expanded_set[node_key].merge_with(node)
                    reach.add_edge(current, expanded_set[node_key])
                    if not exact:
                        open_set1[node_key]=expanded_set[node_key]
                        expanded_set.pop(node_key)
                elif node_key in open_set0:
                    #print "Matched node:\n{}\n\n with node: \n{}".format(node, open_set0[node_key])
                    open_set0[node_key].merge_with(node)
                    reach.add_edge(current, open_set0[node_key])
                elif node_key in open_set1:
                    #print "Matched node:\n{}\n\n with node: \n{}".format(node, open_set1[node_key])
                    open_set1[node_key].merge_with(node)
                    reach.add_edge(current, open_set1[node_key])
                else:
                    if time < steps:
                        open_set1[node_key]=node
                    reach.add_node(node)
                    reach.add_edge(current, node)
        if time == steps:
            rospy.logwarn("Terminating early; ran out of steps")
        return reach



    def run_test(self,steps, prune = True):
        ip = copy.deepcopy(self._ip)
        ip.detectors_off()

        pdef = ip._n._def
        state = ip._n._state
        countdown = ip._n._countdown
        times = [ip._n._time]
        
        detectors = {p:{"ok": (ip._n._def.pl_idx("?ok+"+p),ip._detectors[p]["ok"]),
                      "nok": (ip._n._def.pl_idx("?nok+"+p),ip._detectors[p]["nok"])} for p in ip._detectors}


        action_idxs = [ip._n._def.tr_idx(a) for a in ip._actions]
        root=PetriSimNode(state, countdown, pdef,detectors, action_idxs,times)

        if prune:
            return self.run_test_prune(steps, root)
        else:
            return self.run_test_no_prune(steps, root)

    def det_test(self, steps, tiebreak_conflicts = False, rate=1):

        pdef = copy.deepcopy(self._ip._n._def)


        def toticks(secs):
            return max(1,int(rate*secs))

        def get_time(act, det, which):
            print "Getting time: ", act, det, which
            return 1

        adj_trs = []
        
        for d in self._ip._detectors:
            det_pls = ["?ok+"+d, "?nok+"+d]
            for pl in det_pls:
                det_trs = pdef.get_pl_outputs(pl)[1]
                for tr in det_trs:
                    pdef.rm_tr_arc(tr, pl, out = True)
                    pdef.rm_tr_arc(tr, pl, out = False)
                    adj_trs.append(tr)

        tr_ranges = []
                    
        for tr in adj_trs:
            act, det, which = tr.split("+")


            if which == "got":
                timeout_pls = filter(lambda s: "timeout" in s, pdef.get_pl_outputs("{}+{}+get".format(act, det))[1])

                if len(timeout_pls)>0:
                    max_time = 1+ pdef.tr_time(timeout_pls[0])
                else:
                    max_time = None
                min_time = 1

            if which == "interrupt":
                act_time = pdef.tr_time(act)
                min_time = pdef.tr_time(tr)
                max_time = act_time+1

            tr_ranges.append(range(min_time,max_time+1))

        out = {}
            
        for combos in itertools.product(*tr_ranges):
            print "="*50
            print combos
            for i in range(len(adj_trs)):
                tr = adj_trs[i]
                time = combos[i]
                pdef.set_tr_time(tr, time)
            status, looped, nsteps, conflicts, actions, interrupts, timeouts,completions =  self.simple_test(steps, tiebreak_conflicts, pdef)
            
            print nsteps, status, looped
            print "-"*20, "actions", "-"*20
            print completions
            print "-"*20, "interrupts", "-"*20
            print interrupts
            print "-"*20, "timeouts", "-"*20
            print "-"*20, "conflicts", "-"*20
            print conflicts
            out[combos]=copy.deepcopy((status, looped, nsteps, conflicts, actions, interrupts, timeouts,completions))
        return out

        
    def simple_test(self, steps, tiebreak_conflicts = False, pdef = None):
        if pdef == None:
            pdef = self._ip._n._def
        state = copy.deepcopy(self._ip._n._state)
        countdown = copy.deepcopy(self._ip._n._countdown)

        status = None
        net_ok = True
        onstep=0
        seen_states = set([])
        prev_active = []
        action_history = []
        interrupt_history = []
        timeout_history = []
        conflict_history = []
        action_completions = []

        looped = False
        time = 0
        while net_ok and onstep < steps:
            status = pdef.evaluate(state,countdown,None, tiebreak_conflicts)
            state = status[0]
            time += status[4]
            countdown = status[1]
            if status[8]!=None:
                actions = filter(lambda s: s!= None and  "+" not in s, status[8])
                if len(actions)>0:
                    for a in actions:
                        action_completions.append((time, a))

            net_ok = status[3]=="ok" or status[3]=="interrupt" or (tiebreak_conflicts and status[3]=="conflict")

            if status[3]=="interrupt":
                interrupted = filter(lambda s: not "+timeout" in s[0],status[7])
                if len(interrupted) > 0:
                    interrupt_history.append((time, interrupted))

            if len(status[9])>0:
                conflict_history.append((time,status[9]))

            active = filter(lambda s: not "+" in s, status[6])
            if active != prev_active:
                action_history.append((time,active))
            prev_active = active

            timeouts = filter(lambda s: "+timeout" in s, status[8])
            if len(timeouts)>0:
                timeout_history.append((time,timeouts))

            tuplestate = tuple(np.ndarray.flatten(state).tolist()+np.ndarray.flatten(countdown).tolist())
            loop = tuplestate in seen_states
            if loop:
                looped = True
                break
            seen_states.add(tuplestate)
            onstep+=1

        return status[3], looped, onstep, conflict_history, action_history, interrupt_history, timeout_history, action_completions
