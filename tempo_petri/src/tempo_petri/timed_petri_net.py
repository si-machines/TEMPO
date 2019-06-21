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

import numpy as np
from scipy import sparse
import copy
import itertools
import networkx as nx
import random

class PetriNetDefinition():
    def __init__(self):
        self._trs=[]
        self._pls=[]
        self._a_minus = np.array([], dtype=np.int)
        self._a_plus = np.array([], dtype=np.int)
        self._tr_times={}
        self._tr_values=[]
        self._sparse_at = None
        self._sparse_a_minust = None

    def get_tr_time(self, tr_name):
        return self._tr_times(tr_name)
        
    def add_transition(self, name, inputs=[], outputs=[], time=1, value=1):
        if name not in self._trs:
            self._trs.append(name)
            self._tr_times[name] = time
            self._tr_values.append(value)
        else:
            raise ValueError("Transition name {} already exists".format(name))

        if self._sparse_at != None or self._sparse_a_minust != None:
            self._sparse_at = None
            self._sparse_a_minust = None

        
        if len(self._pls)==0:
            return

        
        minus = np.zeros((1,len(self._pls)), dtype=np.int)
        plus = np.zeros((1,len(self._pls)), dtype=np.int)
        
        if len(self._trs)==1: #this is first transition added
            self._a_minus = minus
            self._a_plus = plus
        else:
            self._a_minus = np.append(self._a_minus, minus, axis=0)
            self._a_plus = np.append(self._a_plus, plus, axis=0)


        for iname in inputs:
            tr_idx = self._trs.index(name)
            pl_idx = self._pls.index(iname)
            self._a_minus[tr_idx][pl_idx] += 1

        for oname in outputs:
            tr_idx = self._trs.index(name)
            pl_idx = self._pls.index(oname)
            self._a_plus[tr_idx][pl_idx] += 1

    def add_place(self, name, inputs=[], outputs=[]):
        if name not in self._pls:
            self._pls.append(name)
        else:
            raise ValueError("Place name {} already exists".format(name))

        if self._sparse_at != None or self._sparse_a_minust != None:
            self._sparse_at = None
            self._sparse_a_minust = None

        
        
        if len(self._trs)==0:
            return

        minus = np.zeros((len(self._trs),1), dtype=np.int)
        plus = np.zeros((len(self._trs),1), dtype=np.int)
        if len(self._pls)==1: #this is the first place added
            self._a_minus = minus
            self._a_plus =  plus
        else:
            self._a_minus = np.append(self._a_minus, minus, axis=1)
            self._a_plus =  np.append(self._a_plus, plus, axis=1)

        for iname in inputs:
            tr_idx = self._trs.index(iname)
            pl_idx = self._pls.index(name)
            self._a_plus[tr_idx][pl_idx] += 1
            
        for oname in outputs:
            tr_idx = self._trs.index(oname)
            pl_idx = self._pls.index(name)
            self._a_minus[tr_idx][pl_idx] += 1
    
    
    def add_pl_arcs(self, name, inputs=[], outputs=[]):
        for iname in inputs:
            tr_idx = self._trs.index(iname)
            pl_idx = self._pls.index(name)
            self._a_plus[tr_idx][pl_idx] += 1
            
        for oname in outputs:
            tr_idx = self._trs.index(oname)
            pl_idx = self._pls.index(name)
            self._a_minus[tr_idx][pl_idx] += 1

    def rm_tr_arc(self, trname, plname, out = False):
        tr_idx = self._trs.index(trname)
        pl_idx = self._pls.index(plname)

        if out:
            if self._a_plus[tr_idx][pl_idx] > 0:
                self._a_plus[tr_idx][pl_idx] -= 1
            else:
                raise ValueError("No arc exists!")
        else:
            if self._a_minus[tr_idx][pl_idx] > 0:
                self._a_minus[tr_idx][pl_idx] -= 1
            else:
                raise ValueError("No arc exists!")
        
    def rm_pl_arc(self, plname, trname, out = False):
        tr_idx = self._trs.index(trname)
        pl_idx = self._pls.index(plname)

        if not out:
            if self._a_plus[tr_idx][pl_idx] > 0:
                self._a_plus[tr_idx][pl_idx] -= 1
            else:
                raise ValueError("No arc exists!")
        else:
            if self._a_minus[tr_idx][pl_idx] > 0:
                self._a_minus[tr_idx][pl_idx] -= 1
            else:
                raise ValueError("No arc exists!")
        
            
    def add_tr_arcs(self, name, inputs=[], outputs=[]):
        for iname in inputs:
            tr_idx = self._trs.index(name)
            pl_idx = self._pls.index(iname)
            self._a_minus[tr_idx][pl_idx] += 1
           
        for oname in outputs:
            tr_idx = self._trs.index(name)
            pl_idx = self._pls.index(oname)
            self._a_plus[tr_idx][pl_idx] += 1


        
    def test_firing(self, firing, firing_result, state):
        conflicts = firing_result>state

        conflicted = []
        for i in range(len(conflicts)):
            if conflicts[i]:
                conflicters = []
                for j in range(len(firing)):
                    if firing[j] and self._a_minus[j,i]>0:
                        conflicters.append(self._trs[j])
                conflicted.append((self._pls[i], conflicters, state[i,0]))
        return conflicted

    def evaluate(self, state, last_countdown, timestep, random_tiebreak=False):
        next_state = copy.deepcopy(state)
        countdown = copy.deepcopy(last_countdown)

        countdown, interrupted_first=self.advance_time(next_state, countdown, 0)

        enabled = countdown[countdown>0]
        enabled_names = [self._trs[i] for i in range(len(countdown)) if countdown[i,0]>0]

        in_conflict = []

        if len(enabled)==0:
            self._stopped=True
            return next_state, countdown, countdown==False, "dead", 0,enabled_names,[],[],[],[]
        else:
            if timestep==None:
                timestep=min(enabled)
            else:
                if timestep>min(enabled):
                    raise ValueError("Step must be less than time to next event")

        elapsed_time=timestep
        firing = countdown==timestep

        if self._sparse_at==None:
            self._sparse_at = sparsemat = sparse.csr_matrix(np.transpose(self._a_plus))-sparse.csr_matrix(np.transpose(self._a_minus))
        if self._sparse_a_minust==None:
            self.sparse_a_minust=sparse.csr_matrix(np.transpose(self._a_minus))

        firing_needs = self.sparse_a_minust.dot(sparse.csr_matrix(firing)).todense()
        firing_result = self._sparse_at.dot(sparse.csr_matrix(firing)).todense()
        conflicted = self.test_firing(firing, firing_needs, state)
    

        if len(conflicted)>0:
            resolved = []
            rand_resolved = []
            all_dont_fire=[]
            rand_selection = {}
            
            for pl,trs,avail in conflicted:
                rand_selection[pl]=[]
                tr_idxs = map(lambda tr: self._trs.index(tr),trs)
                tr_vals = [self._tr_values[idx] for idx in tr_idxs]
                pl_idx = self._pls.index(pl)
                tr_costs = [self._a_minus[idx, pl_idx] for idx in tr_idxs]

                sorted_trs = sorted(zip(tr_vals,tr_idxs,tr_costs),reverse=True)
                
                avail = state[pl_idx,0]
                dont_fire = []

                solved = True
                best_trs = []
                while avail > 0 and len(sorted_trs)>0:
                    current_val = sorted_trs[0][0]
                    best_trs = []
                    while len(sorted_trs)>0 and sorted_trs[0][0]==current_val:
                        best_trs.append(sorted_trs.pop(0))
                    if sum(map(lambda g: g[2], best_trs)) > avail:
                        if current_val <= avail:
                            if random_tiebreak:
                                while current_val <= avail:
                                    sel = best_trs.pop(random.randint(0,len(best_trs)-1))
                                    rand_selection[pl].append(self._trs[sel[1]])
                                    avail-=current_val
                                for sel in best_trs:
                                    dont_fire.append(sel[1])
                                rand_resolved.append(pl)
                            else:
                                solved = False
                                break
                        else:
                            for sel in best_trs:
                                dont_fire.append(sel[1])
                    else:
                        avail-=len(best_trs)*current_val

                for sel in sorted_trs:
                    dont_fire.append(sel[1])

                if len(dont_fire)==len(tr_idxs):
                    solved = False
                else:
                    all_dont_fire.append(dont_fire)
                
                if solved:
                    resolved.append(pl)

            
            if len(resolved)<len(conflicted):
                in_conflict=[(s[0],s[1],s[2],rand_selection[s[0]]) for s in conflicted if s[0] not in resolved]
                self._stopped=True
                return next_state, countdown, firing, "conflict", 0,enabled_names, [], [], [], in_conflict
            else:
                in_conflict=[(s[0],s[1],s[2],rand_selection[s[0]]) for s in conflicted if s[0] in rand_resolved]
                for tr_idx in all_dont_fire:
                    firing[tr_idx,0]=False
            firing_needs = self.sparse_a_minust.dot(sparse.csr_matrix(firing)).todense()
            firing_result = self._sparse_at.dot(sparse.csr_matrix(firing)).todense()

        next_state = next_state+np.asarray(firing_result)

        countdown, interrupted=self.advance_time(next_state, countdown, timestep)

        interrupted+=interrupted_first

        if len(interrupted)==0:
            status = "ok"
        else:
            status = "interrupt"

        enabled = countdown[countdown>0]
        enabled_after = [self._trs[i] for i in range(len(countdown)) if countdown[i,0]>0]
        outfiring = np.ndarray.flatten(firing*1).tolist()
        outfiring = zip(self._trs, outfiring)
        outfiring = filter(lambda p: p[1]>0, outfiring)
        all_fired = map(lambda p: p[0], outfiring)

        return next_state, countdown, firing, status, timestep, enabled_names, enabled_after, interrupted, all_fired, in_conflict

    def places(self):
        return tuple(self._pls)

    def transitions(self):
        return tuple(self._trs)

    def pl_idx(self, place):
        return self._pls.index(place)

    def tr_idx(self,place):
        return self._trs.index(place)


    def tr_time(self, name):
        if type(self._tr_times[name])==int:
            return self._tr_times[name]
        else:
            return self._tr_times[name]()

    def set_tr_time(self, name, time):
        if name in self._tr_times:
            self._tr_times[name]=time

    def advance_time(self,state, countdown, timestep):
        def advance_time(c):
            if c > 0:
                return c-timestep
            else:
                return c
        
        new_countdown = np.array(map(advance_time, countdown))


        interrupted = []
        enabled = [a.all() for a in (np.transpose(state)>=self._a_minus)]
        for i in range(len(enabled)):
            if enabled[i]:
                if new_countdown[i,0]<=0:
                    new_countdown[i,0]=self.tr_time(self._trs[i])
            else:
                if new_countdown[i,0]>0:
                    interrupted.append((self._trs[i],new_countdown[i,0]))
                    new_countdown[i,0]=0
        return new_countdown, interrupted



    def get_tr_inputs(self,name):
        if name==None:
            return np.sum(self._a_minus, axis=1).tolist()
        tr_idx = self._trs.index(name)
        
        inputs = self._a_minus[tr_idx,:]

        num = sum(inputs)
        which = [self._pls[i] for i in range(len(inputs)) if inputs[i]>0]

        return (num, which)

    def get_tr_outputs(self,name=None):
        if name==None:
            return np.sum(self._a_plus, axis=1).tolist()
        tr_idx = self._trs.index(name)
        
        outputs = self._a_plus[tr_idx,:]

        num = sum(outputs)
        which = [self._pls[i] for i in range(len(outputs)) if outputs[i]>0]

        return (num, which)

    def get_pl_inputs(self, name = None):
        if name==None:
            return np.sum(self._a_plus, axis=0).tolist()

        pl_idx = self._pls.index(name)
        
        inputs = self._a_plus[:,pl_idx]

        num = sum(inputs)
        which = [self._trs[i] for i in range(len(inputs)) if inputs[i]>0]

        return (num, which)

    def get_pl_outputs(self, name=None):
        if name==None:
            return np.sum(self._a_minus, axis=0).tolist()

        pl_idx = self._pls.index(name)
        
        outputs = self._a_minus[:,pl_idx]

        num = sum(outputs)
        which = [self._trs[i] for i in range(len(outputs)) if outputs[i]>0]

        return (num, which)



class PetriNet:
    def __init__(self):
        self._def = PetriNetDefinition()
        self._countdown = np.array([], dtype=np.int)
        self._state = np.array([], dtype=np.int)
        self._time = None
        self._stopped = None
        self._in_conflict = []
        self._history = np.array([], dtype=np.int)
        self._empty_idx = 0
        self._graph = nx.DiGraph()

        
    def add_transition(self, name, inputs=[], outputs=[], time=1, value=1):
        self._def.add_transition(name, inputs, outputs, time, value)
        self._graph.add_node("tr:"+name, label="tr:{}\n{}".format(time, name), shape="box")


        countdown = np.zeros((1,1), dtype=np.int)
        if len(self._def.transitions())==1: #this is first transition added
            self._countdown = countdown
        else:
            self._countdown = np.append(self._countdown, countdown, axis = 0) 

        for iname in inputs:
            self._graph.add_edge("pl:"+iname, "tr:"+name)

        for oname in outputs:
            self._graph.add_edge("tr:"+name,"pl:"+oname)

    def add_place(self, name, inputs=[], outputs=[], tokens=0):
        self._def.add_place(name, inputs, outputs)
        state = np.zeros((1,1),dtype=np.int)
        state[0,0]=tokens
        if len(self._def._pls)==1:
            self._state  = state
        else:
            self._state = np.append(self._state, state, axis=0)
        self._graph.add_node("pl:"+name, label = "pl:{}\n{}".format("", name))
        for iname in inputs:
            self._graph.add_edge("tr:"+iname, "pl:"+name)
            
        for oname in outputs:
            self._graph.add_edge("pl:"+name,"tr:"+oname)

    def add_pl_arcs(self, name, inputs=[], outputs=[]):
        self._def.add_pl_arcs(name, inputs, outputs)
        for iname in inputs:
            self._graph.add_edge("tr:"+iname, "pl:"+name)
            
        for oname in outputs:
            self._graph.add_edge("pl:"+name,"tr:"+oname)

    def add_tr_arcs(self, name, inputs=[], outputs=[]):
        self._def.add_tr_arcs(name, inputs, outputs)
        for iname in inputs:
            self._graph.add_edge("pl:"+iname, "tr:"+name)
           
        for oname in outputs:
            self._graph.add_edge("tr:"+name,"pl:"+oname)

    def stopped(self):
        return self._stopped

    def initialize(self):
        self._time = 0
        self._stopped = False
        self._history = np.zeros_like(self._countdown)
        self._countdown, interrupted = self._def.advance_time(self._state,self._countdown, 0)


    def get_tr_time(self, name):
        return self._def.tr_time(name)

    def set_tr_time(self, name, time):
        if self._countdown[self._def._trs.index(name),0]>0:
            rospy.logwarn("Changing time of active transition; change will not take effect until next firing")
        self._def.set_tr_time(name,time)

    def step(self, timestep=None):
        results = self._def.evaluate(self._state, self._countdown, timestep, random_tiebreak=True)
        self._state = results[0]
        self._countdown = results[1]
        firing = results[2]

        if len(self._history)==0:
            self._history=firing*1
        else:
            self._history+=firing*1

        self._time += results[4]
        return results[3:]

    def get_history(self):
        return dict(zip(self._def._trs, list(np.ndarray.flatten(self._history))))


    def set_state(self, state): #state is dict of names & # tokens
        new_state = np.zeros_like(self._state)
        for place_name in self._def.places():
            pl_idx = self._def.pls_idx(place_name)
            new_state[pl_idx,0]=state[place_name]
            token_str = "O"*new_state[pl_idx,0]
            nx.set_node_attributes(self._graph, "label", {"pl:"+place: "pl:{}\n{}".format(token_str, place)})
        self._state=new_state
        self._countdown, interrupted = self._def.advance_time(self._state,self._countdown, 0)
        return interrupted

    def get_state(self):
        return dict(zip(self.places(), [int(n) for n in np.ndarray.flatten(self._state).tolist()]))

    def get_tr_inputs(self,name):
        return self._def.get_tr_inputs(name)

    def get_tr_outputs(self,name=None):
        return self._def.get_tr_outputs(name)

    def get_pl_inputs(self, name = None):
        return self._def.get_pl_inputs(name)

    def get_pl_outputs(self, name=None):
        return self._def.get_pl_outputs(name)

    def places(self):
        return self._def.places()

    def transitions(self):
        return self._def.transitions()

    def set_tokens(self, place, tokens):
        pl_idx = self._def.pl_idx(place)
        self._state[pl_idx,0]=tokens
        token_str = "O"*self._state[pl_idx,0]
        nx.set_node_attributes(self._graph, "label", {"pl:"+place: "pl:{}\n{}".format(token_str, place)})
        self._countdown, interrupted = self._def.advance_time(self._state,self._countdown, 0)
        return interrupted

    def add_token(self, place):
        pl_idx = self._def.pl_idx(place)
        self._state[pl_idx,0]+=1
        token_str = "O"*self._state[pl_idx,0]
        nx.set_node_attributes(self._graph, "label", {"pl:"+place: "pl:{}\n{}".format(token_str, place)})
        self._countdown, interrupted = self._def.advance_time(self._state,self._countdown, 0)
        return interrupted
        
    def history_repr(self):
        s = ""
        s += "\nHistory:"
        for i in range(len(self._trs)):
            if self._history[i,0]>0:
                s+="\n  {:>3} {}".format( self._history[i,0],self._trs[i])

        return s

    def get_nx_graph(self):
        return self._graph
                
    def __repr__(self):
        s = "*"*30

        '''s+="\nArcs:\n{}".format(self._a_plus-self._a_minus)
        s+="\nState:"
        for i in range(len(self._pls)):
            s+="\n  {:>3} {}".format(self._state[i,0],self._pls[i])
        s+="\nCountdown:"
        for i in range(len(self._trs)):
            s+="\n  {:>3} {}".format( self._countdown[i,0],self._trs[i])'''

        s+="\nState:"
        for i in range(len(self._def.places())):
            if self._state[i,0]>0:
                s+="\n  {:>3} {}".format(self._state[i,0],self._def.places()[i])
        s+="\nCountdown:"
        for i in range(len(self._def.transitions())):
            if self._countdown[i,0]>0:
                s+="\n  {:>3} {}".format( self._countdown[i,0],self._def.transitions()[i])

        return s

class InteractionPetri:
    RESTART = 0
    GIVE_UP = 1
    WAIT = 2
    INTERRUPT = 3

    def __init__(self, detectors, resource_pools): #detectors: list, resource_pools: dict w/ # items
        for d in detectors:
            if "+" in d:
                raise ValueError("'+' character reserved; not allowed in petri net names")
        for p in resource_pools:
            if "+" in p:
                raise ValueError("'+' character reserved; not allowed in petri net names")
        
            
        self._n = PetriNet()
        self._detectors = dict([(d,{"ok":0,"nok":0}) for d in detectors])
        self._detector_on = {}
        for d in detectors:
            self._detector_on[d]=False
            self._n.add_place("?ok+"+d)
            self._n.add_place("?nok+"+d)

        self._pools = resource_pools.keys()
        for p in resource_pools:
            self._n.add_place("pool+"+p, tokens = resource_pools[p])
            self._n.add_place("pool_intreq+"+p)
            self._n.add_place("pool_cancelint+"+p)
            self._n.add_transition("pool_handle_int+"+p, inputs=["pool_intreq+"+p])
            self._n.add_transition("pool_handle_cancel+"+p, inputs=["pool_cancelint+"+p])

        self._groups = {}
        self._actions = []

    def get_state_no_det(self):
        return [num for f,num in self._n.get_state().items() if f[0:4]=="?nok+" or f[0:3]=="?ok+"]

    def get_detectors(self):
        return copy.deepcopy(self._detectors.keys())

    def get_detector_states(self):
        return copy.deepcopy(self._detector_on)

    def start(self):
        self._n.initialize()

    def flip_detector(self,detector):
        if self._detector_on[detector]:
            self.detector_off(detector)
        else:
            self.detector_on(detector)
    
    def detectors_off(self):
        for d in self._detectors.keys():
            self.detector_off(d)


    def detector_on(self,detector):
        self._n.set_tokens("?nok+"+detector, 0)
        self._n.set_tokens("?ok+"+detector, self._detectors[detector]["ok"])
        self._detector_on[detector]=True

    def detector_off(self,detector):
        self._n.set_tokens("?nok+"+detector, self._detectors[detector]["nok"])
        self._n.set_tokens("?ok+"+detector, 0)
        self._detector_on[detector]=False
        

    def add_result(self, action, result_action):
        self._n.add_tr_arcs(action, outputs=[result_action+"+intent"])

    def add_emptier(self, transition, place):
        name = place+transition+"+empty"
        i=1
        while name in self._n.places():
            name = place+transition+"+empty"+str(i)
            i+=1
        self._n.add_place(place+transition+"+empty"+str(i), inputs=[transition])
        self._n.add_transition(place+transition+"+emptier"+str(i),inputs=[place+transition+"+empty"+str(i), place], value=2)
        self._n.add_transition(place+transition+"+cancel_empty"+str(i), inputs=[place+transition+"+empty"+str(i)])

    def add_limiter(self, place):
        self._n.add_transition(place+"+limiter", inputs=[place,place],outputs=[place], value=0)

    
    def add_action_monitor(self, action, group_name, thread_idx, action_idx):
        name = "{}+{}+thr{}+act{}".format(group_name, action, thread_idx,action_idx)
        in_name = name+"+monitor_begin"
        out_fail_name = name+"+handle_failure"
        out_ok_name = name+"+handle_ok"

        self._n.add_place(in_name)
        self._n.add_place(name+"+fail")
        self._n.add_place(name+"+ok")
        self._n.add_place(name+"+waiting")       

        self._n.add_transition(name+"+start", inputs=[in_name], outputs=[action+"+intent", name+"+waiting"])
        self._n.add_tr_arcs(action+"+handle_failure", outputs=[name+"+fail"])
        self._n.add_tr_arcs(action, outputs=[name+"+ok"])
        self._n.add_transition(out_ok_name, inputs=[name+"+ok", name+"+waiting"])
        self._n.add_transition(out_fail_name, inputs=[name+"+fail", name+"+waiting"])
        self.add_emptier(name+"+start", name+"+ok")
        self.add_emptier(name+"+start", name+"+fail")
        self.add_limiter(name+"+ok")
        self.add_limiter(name+"+fail")
        

        return in_name, out_ok_name, out_fail_name


    def change_action_group_thread_times(self, name, new_times):
        for i in range(len(new_times)):
            time = new_times[i]
            self._n.set_tr_time("{}+thr{}_start".format(name, i), time)
    
    def add_action_group(self,name,actions, repeat=False, stop_on_interrupt=True):
        if name in self._actions or name in self._groups.keys():
            raise ValueError("Action {} already exists in network!".format(name))

        
        self._actions.append(name)
        self._groups[name]=actions

        self._n.add_place(name+"+intent")
        self._n.add_transition(name+"+start", inputs=[name+"+intent"])

        self._n.add_transition(name)
        self._n.add_place(name+"+all_done")

        #TODO: right now these two are unused
        self._n.add_place(name+"+failed")
        self._n.add_transition(name+"+handle_failure", inputs=[name+"+failed"])

        last_trans = []

        for i in range(len(actions)):
            time = actions[i][0]
            self._n.add_place("{}+thr{}_intent".format(name, i), inputs=[name+"+start"])
            self._n.add_transition("{}+thr{}_start".format(name, i), inputs=["{}+thr{}_intent".format(name, i)], time=time)
            last_ok = "{}+thr{}_start".format(name, i)
            last_fail = None
            for j in range(len(actions[i][1])):
                action = actions[i][1][j]
                in_name, out_ok, out_fail = self.add_action_monitor(action, name, i, j)
                self._n.add_tr_arcs(last_ok, outputs=[in_name])
                if (not stop_on_interrupt) and last_fail!=None:
                    self._n.add_tr_arcs(last_fail, outputs=[in_name])
                elif repeat and last_fail!=None:
                    self._n.add_tr_arcs(last_fail, outputs=[name+"+intent"])
                last_ok = out_ok
                last_fail = out_fail
            last_trans.append(last_ok)
            if not stop_on_interrupt:
                last_trans.append(last_fail)
        
        for tr in last_trans:
            self._n.add_tr_arcs(tr, outputs=[name+"+all_done"])
        for i in range(len(actions)):
            self._n.add_tr_arcs(name, inputs = [name+"+all_done"])
        if repeat:
            self._n.add_tr_arcs(name, outputs=[name+"+intent"])

    def add_alternative_group(self, name, actions, repeat_on_fail = False):
        if name in self._actions or name in self._groups.keys():
            raise ValueError("Action {} already exists in network!".format(name))
        if "+" in name:
            raise ValueError("'+' character reserved; not allowed in petri net names")
        if len(actions)<2:
            raise ValueError("Must provide at least one alternative action!")
            

        self._actions.append(name)
        self._n.add_place(name+"+intent")
        self._n.add_transition(name+"+start", inputs=[name+"+intent"])
        self._n.add_place(name+"+done")
        self._n.add_transition(name, inputs=[name+"+done"])
        self._n.add_place(name+"+failed")
        self._n.add_transition(name+"+handle_failure", inputs=[name+"+failed"])

        last_fail = "{}+start".format(name)
        last_ok = None
        for i in range(len(actions)):
            action = actions[i]
            in_name, out_ok, out_fail = self.add_action_monitor(action, name, 0, i)
            self._n.add_tr_arcs(last_fail, outputs=[in_name])
            self._n.add_tr_arcs(out_ok, outputs=[name+"+done"])
            self.add_emptier(last_fail, name+"+done")
            self.add_emptier(last_fail, name+"+failed")
            last_ok = out_ok
            last_fail = out_fail

        if repeat_on_fail:
            self._n.add_tr_arcs(last_fail, outputs = [name+"+intent"])
        else:
            self._n.add_tr_arcs(last_fail, outputs = [name+"+failed"])

        self.add_limiter(name+"+done")
        self.add_limiter(name+"+failed")



    def do_action(self, name):
        if name not in self._groups.keys() and name not in self._actions:
            raise ValueError("Action {} does not exist".format(name))

        self._n.add_token(name+"+intent")

    def add_action(self, name, time, resources, detectors, resource_timeouts=None, detector_timeouts=None,
                   interrupt_delay=1, interrupt_window=0, interrupt_others=2, interrupt_handler=1, resource_dests={}):
        if name in self._actions or name in self._groups.keys():
            raise ValueError("Action {} already exists in network!".format(name))

        if "+" in name:
            raise ValueError("'+' character reserved; not allowed in petri net names")

        self._actions.append(name)
        self._n.add_place(name+"+intent")
        self._n.add_transition(name, time = time)
        
        if len(resources+detectors)==0:
            self._n.add_tr_arcs(name, inputs=[name+"+intent"])
        else:
            self._n.add_transition(name+"+start", inputs=[name+"+intent"])

        self._n.add_place(name+"+failed")
        self._n.add_transition(name+"+handle_failure", inputs=[name+"+failed"])

        for r in resources+detectors:
            pref = name+"+"+r
            self._n.add_place(pref+"+get", inputs=[name+"+start"])
            self._n.add_transition(pref+"+got", inputs=[pref+"+get"])
            self._n.add_place(pref+"+ok", inputs=[pref+"+got"])
            self._n.add_place(pref+"+intern_int")
            self._n.add_transition(pref+"+cleanup", inputs=[pref+"+intern_int",pref+"+ok"])
            self._n.add_tr_arcs(name, inputs=[pref+"+ok"])
            self._n.add_transition(pref+"+interrupt", 
                                   inputs=[pref+"+ok"],
                                   time=interrupt_delay,
                                   value=0)


        for i in range(len(resources)):
            r = resources[i]
            pref = name+"+"+r

            if resource_timeouts and resource_timeouts[i]!=0:
                self._n.add_transition(pref+"+timeout", time = resource_timeouts[i], 
                                       inputs=[pref+"+get"],
                                       outputs=[name+"+failed"], value=0)

            
            if interrupt_others==self.INTERRUPT:
                self._n.add_tr_arcs(name+"+start", outputs=["pool_intreq+"+r])
                self._n.add_tr_arcs(pref+"+got", outputs=["pool_cancelint+"+r])

            self._n.add_tr_arcs(pref+"+got", inputs=["pool+"+r])

            self._n.add_tr_arcs(pref+"+cleanup", outputs=["pool+"+r])
            self._n.add_place(pref+"+extern_int", inputs=["pool_handle_int+"+r])
            self._n.add_place(pref+"+cancel_ext_int", inputs=["pool_handle_cancel+"+r])
            self._n.add_transition(pref+"+handle_cancel_int", 
                                   inputs=[pref+"+extern_int",pref+"+cancel_ext_int"])

            self._n.add_tr_arcs(pref+"+interrupt", 
                                inputs=[pref+"+extern_int"], 
                                outputs=["pool+"+r])


        for i in range(len(detectors)):
            d = detectors[i]
            pref = name+"+"+d

            if detector_timeouts and detector_timeouts[i]!=0:
                self._n.add_transition(pref+"+timeout", time=detector_timeouts[i],
                                       inputs=[pref+"+get"],
                                       outputs=[name+"+failed"], value=0)

            self._detectors[d]["ok"]+=1
            self._detectors[d]["nok"]+=1

            self._n.add_tr_arcs(pref+"+got", inputs=["?ok+"+d], outputs=["?ok+"+d])
            self._n.add_tr_arcs(pref+"+interrupt", 
                                inputs=["?nok+"+d],
                                outputs=["?nok+"+d])

        for r in resource_dests:
            self._n.add_tr_arcs(name, outputs=["pool+"+resource_dests[r]])

        for i in range(len(resources)):
            r = resources[i]
            if r not in resource_dests:
                self._n.add_tr_arcs(name, outputs=["pool+"+r])

            if resource_timeouts and resource_timeouts[i]!=0:
                self._n.add_tr_arcs(name+"+"+r+"+timeout", 
                                    outputs=[name+"+"+r2+"+intern_int" 
                                             for r2 in resources+detectors if r!=r2])
            
        for i in range(len(detectors)):
            r = detectors[i]
            if detector_timeouts and detector_timeouts[i]!=0:
                self._n.add_tr_arcs(name+"+"+r+"+timeout", 
                                    outputs=[name+"+"+r2+"+intern_int" 
                                             for r2 in resources+detectors if r!=r2])

        if interrupt_window > 0:
            resource_trs = ["{}+{}+got".format(name, d) for d in resources+detectors]
            self._n.add_place(name+"+interrupt_ready")
            self._n.add_pl_arcs(name+"+interrupt_ready", inputs=resource_trs)
            self._n.add_place(name+"+interruptible")
            self._n.add_transition(name+"+start_interrupt_window",
                                   inputs=[name+"+interrupt_ready"]*len(resources+detectors),
                                   outputs=[name+"+interruptible"])
            self._n.add_transition(name+"+end_interrupt_window",
                                   inputs = [name+"+interruptible"],
                                   time=interrupt_window)

                
        for r in resources+detectors:
            if interrupt_window > 0:
                self._n.add_tr_arcs("{}+{}+interrupt".format(name,r),
                                    inputs=["{}+interruptible".format(name)])
            if interrupt_handler==self.RESTART or interrupt_handler==self.GIVE_UP:
                self._n.add_tr_arcs(name+"+"+r+"+interrupt", 
                                    outputs=[name+"+"+r2+"+intern_int" 
                                             for r2 in resources+detectors if r!=r2])
            if interrupt_handler==self.GIVE_UP:
                self._n.add_tr_arcs(name+"+"+r+"+interrupt", 
                                    outputs=[name+"+failed"])
            if interrupt_handler==self.RESTART:
                self._n.add_tr_arcs(name+"+"+r+"+interrupt",
                                    outputs=[name+"+intent"])
            if interrupt_handler==self.WAIT:
                self._n.add_tr_arcs(name+"+"+r+"+interrupt",
                                    outputs=[name+"+"+r+"+get"])

       
                
        for r in resources+detectors:
            pref = name+"+"+r
            self.add_limiter(pref+"+intern_int")
            self.add_emptier(name+"+start",pref+"+intern_int")
            if r in resources:
                self.add_limiter(pref+"+extern_int")
                self.add_limiter(pref+"+cancel_ext_int")
                self.add_emptier(pref+"+got", pref+"+extern_int")
                self.add_emptier(pref+"+interrupt", pref+"+cancel_ext_int")
                self.add_emptier("pool_handle_int+"+r, pref+"+cancel_ext_int")


    def step(self,timestep=None):
        return self._n.step(timestep)

    def get_pool_state(self):
        return dict([(f[5:],num) for f,num in self._n.get_state().items() if f in map(lambda s: "pool+"+s, self._pools)])
    def get_started_state(self):
        return dict([(f,num) for f,num in self._n.get_state().items() if "+ok" in f])

    def get_nz_intent_state(self):
        return dict([(f,num) for f,num in self._n.get_state().items() if "+intent" in f and num > 0])

    def get_nz_started_state(self):
        return dict([(f,num) for f,num in self._n.get_state().items() if "+ok" in f and num > 0])

    def run_to_end(self, maxsteps=100):
        i = 0
        while i<maxsteps and not self._n.stopped():
            res = self._n.step()
            i+=1

    def __repr__(self):
        s = "="*30
        s += "\nPools:"
        for pool, ntokens in self.get_pool_state().items():
            s+="\n  {:>3} {}".format(ntokens,pool)
        s += "\nActive Actions:"
        for countdown, a in self.get_active_actions():
            s+="\n  {:>3} {}".format(countdown,a)
        s += "\nWaiting Actions:"
        for countdown, a, r in self.get_waiting_actions():
            s+="\n  {:>3} {} ({})".format(countdown,a,r)

        #return s+"\n"+self._n.__repr__()
        return s

    def get_waiting_actions(self):
        ret = []
        for a in self._actions:
            for r in self._pools+self._detectors.keys():
                if "{}_{}+timeout".format(a,r) in self._n.transitions():
                    countdown = self._n._countdown[self._n.transitions().index("{}_{}+timeout".format(a,r)),0]
                    if countdown > 0:
                        ret.append((countdown, a, r))
        return ret

    def get_active_actions(self):
        ret = []
        for a in self._actions:
            countdown = self._n._countdown[self._n.transitions().index(a),0]
            if countdown > 0:
                ret.append((countdown,a))
        return ret

    def get_history(self):
        return copy.deepcopy(self._n._history)

    def stopped(self):
        return self._n.stopped()
    
    def get_state(self):
        return copy.deepcopy(np.ndarray.flatten(self._n._state).tolist())

    def get_countdown(self):
        return copy.deepcopy(np.ndarray.flatten(self._n._countdown).tolist())

    def get_places(self):
        return copy.deepcopy(self._n.places())

    def get_trs(self):
        return copy.deepcopy(self._n.transitions())

    def get_count_repr(self):
        s = ""
        s += "\nCountdown:"
        trs = self.get_trs()
        
        for i in range(len(trs)):
            if self._n._countdown[i,0]>0:
                s+="\n  {:>3} {}".format(self._n._countdown[i,0],trs[i])
        return s

    def get_history_repr(self):
        return self._n.history_repr()

    def get_state_repr(self):
        s = ""
        s += "\nState:"
        places = self.get_places()
        
        for i in range(len(places)):
            if self._n._state[i,0]>0:
                s+="\n  {:>3} {}".format(self._n._state[i,0],places[i])
        return s

    def get_time(self):
        return copy.deepcopy(self._n._time)

class PetriSimNode2:
    def __init__(self):
        pass

    def get_branches(self):
        pass

    def best_case_branch(self):
        pass

    def get_next(self, petri, step1=False):
        pass

    def get_time(self):
        pass

    def get_state(self):
        pass
    
    def get_count(self):
        pass

    def get_tokens(self, capped=False, include_detectors=True):
        pass

    def capped_eq(self,other):
        pass

    def no_detector_eq(self,other):
        pass

    def capped_hash(self, other):
        pass

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        if self.get_state()==other.get_state():
            return True
        return False
        

    def __neq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return not self.__eq__(other)
    
    def __hash__(self):
        pass

    def get_active(self):
        pass
        
    def last_changes(self):
        pass

    def __repr__(self):
        pass

    def history(self):
        pass

    


class PetriSimNode:
    def __init__(self, state, countdown, petri_definition, detectors, action_idxs, times = [0]):
        self._petri = petri_definition
        self.last_status = None
        self.times = times
        self.state = state
        self.countdown = countdown
        self.detector_dict = detectors
        self.detectors = detectors.values()
        self.action_idxs = action_idxs
        self.countdowns = [countdown]
        
        self.det_idxs = set([])
        for d in self.detectors:
            for det_info in d.values():
                self.det_idxs.add(det_info[0])

        self.det_depend_place_idxs={}
        for pl_idx in range(len(self._petri._pls)):
            for name in self.detector_dict:
                if name+"+get" in  self._petri._pls[pl_idx] or name+"+ok" in self._petri._pls[pl_idx]:
                    self.det_depend_place_idxs[name]=pl_idx

                
        
    def flip_detector(self, det_idx):
        if self.state[self.detectors[det_idx]["ok"][0]]==0:
            self.state[self.detectors[det_idx]["ok"][0]] = self.detectors[det_idx]["ok"][1]
            self.state[self.detectors[det_idx]["nok"][0]] = 0
        else:
            self.state[self.detectors[det_idx]["nok"][0]] = self.detectors[det_idx]["nok"][1]
            self.state[self.detectors[det_idx]["ok"][0]] = 0

    def detector_off(self, det_idx):
        self.state[self.detectors[det_idx]["nok"][0]] = self.detectors[det_idx]["nok"][1]
        self.state[self.detectors[det_idx]["ok"][0]] = 0

    def detector_on(self, det_idx):
        self.state[self.detectors[det_idx]["nok"][0]] = 0
        self.state[self.detectors[det_idx]["ok"][0]] = self.detectors[det_idx]["nok"][1]
        
    def step(self):
        self.last_status = self._petri.evaluate(self.state, self.countdown, 1)
        self.state = self.last_status[0]
        self.countdown = self.last_status[1]
        self.countdowns[0]=self.countdown

    def det_on_next(self):
        next_steps = []
        detectors = range(len(self.detectors))
        
        #print "#"*40
        n = PetriSimNode(copy.deepcopy(self.state), copy.deepcopy(self.countdown), 
                         self._petri, self.detector_dict, self.action_idxs, copy.deepcopy(self.times))
        
        for d in detectors:
            n.detector_on(detectors[d])            
        
        n.step()
        for d in detectors:
            n.detector_off(d)
        next_steps.append(n)

        next_steps = list(set(next_steps))

    def get_next(self):
        if self.last_status!=None and (self.last_status[3]=="dead" or self.last_status[3]=="conflict"):
            return []

        
        
        
        for combos in itertools.product([True,False],repeat=len(detectors)):
            for d in range(len(detectors)):
                n = PetriSimNode(copy.deepcopy(self.state), copy.deepcopy(countdown), 
                                 self._petri, self.detector_dict, self.action_idxs, copy.deepcopy(self.times))
                if combos[d]:
                    n.flip_detector(detectors[d])            

               
                state = n.get_state()
                #print [s for s in state if s>0 ]
     
                n.step()
                
                state = n.get_state()
                #print [s for s in state if s>0 ]

                if state not in seen:
                    next_steps.append(n)
                    seen.add(state)
        

        
    def get_next(self):
        if self.last_status!=None and (self.last_status[3]=="dead" or self.last_status[3]=="conflict"):
            return []

        next_steps = []
        detectors = range(len(self.detectors))
        seen = set([])

        #print "#"*40
        #print self
        #print "-"*40
        for countdown in self.countdowns:
            #print [i for i in np.ndarray.flatten(countdown).tolist() if i>0]
            for combos in itertools.product([True,False],repeat=len(detectors)):
                for d in range(len(detectors)):
                    n = PetriSimNode(copy.deepcopy(self.state), copy.deepcopy(countdown), 
                                     self._petri, self.detector_dict, self.action_idxs, copy.deepcopy(self.times))
                    if combos[d]:
                        n.flip_detector(detectors[d])            

               
                    state = n.get_state()
                    #print [s for s in state if s>0 ]
     
                    n.step()

                    state = n.get_state()
                    #print [s for s in state if s>0 ]

                    if state not in seen:
                        next_steps.append(n)
                        seen.add(state)

        #print "\n".join(map(lambda s:str(s), next_steps))
        #print "#"*40
        return next_steps
    

    def key(self):
        if self.last_status != None:
            status = [self.last_status[3]]
        else:
            status = ["None"]
        
        

        #return self.get_intervals() + tuple(status)
        return self.get_state()#+tuple(status)
        #return self.get_tokens_no_det()+tuple(status)
        #return self.get_tokens_no_det()+self.get_count()
        
    def get_intervals(self):
        count = self.get_count()
            
        nz = [i for i in count if i > 0]
        if len(nz)>0:
            min_nz = min(nz)
        else:
            min_nz = 0
        intervals = tuple([n-min_nz if n-min_nz>=0 else -1 for n in count])
        return intervals

    def merge_with(self, other):

        eq = True
        for c in other.countdowns:
            found = False
            for d in self.countdowns:
                if all(c==d):
                    found = True
            
            if not found:
                eq = False
                self.countdowns.append(c)
        return eq


    def add_times(self, times):
        self.times+=times
        self.times=list(set(self.times))

    def get_times(self):
        return self.times

    def get_state(self):
        return self.get_tokens_no_det()+self.get_count()

    def get_tokens(self):
        return tuple(map(lambda l: int(l), np.ndarray.flatten(self.state).tolist()))

    def get_tokens_no_det(self):
        state = map(lambda l: int(l), np.ndarray.flatten(self.state).tolist())
        state = [state[i] for i in range(len(state)) if i not in self.det_idxs]
        return tuple(state)

    def get_count(self):
        return tuple(map(lambda l: int(l), np.ndarray.flatten(self.countdown).tolist()))

    def get_active(self):
        ret = []
        for a in self.action_idxs:
            countdown = self.countdown[a,0]
            if countdown > 0:
                ret.append((countdown,self._petri.transitions()[a]))
        return ret
        
    def last_changes(self):
        active_actions = self.get_active()
        action_intents = []#map(lambda s: s.split("+")[0], self._petri.get_nz_intent_state().keys())        
        if self.last_status!=None and self.last_status[3] =="interrupt":
            interrupts = self.last_status[7]
            interrupts = filter(lambda s: not "+" in s[0], interrupts)
            interrupts = map(lambda s: s[0], interrupts)
        elif self.last_status!=None and self.last_status[3]=="conflict":
            interrupts = self.last_status[8]
        else:
            interrupts = []
        return active_actions,action_intents,interrupts

    def __repr__(self):
        active, intents, interrupts = self.last_changes()
        
        if self.last_status != None:
            s = self.last_status[3]+"\n"
        else:
            s = ""
        for time, action in active:
            #s+="{}:{} ".format(action,time)
            s+="*{}* ".format(action)

        for action in intents:
            s+="{}:{} ".format(action.split("+")[0],"st")

        s+="\n"
        for action in interrupts:
            #s+="{}:{} ".format(action.split("+")[0],"int")
            if self.last_status[3]=="interrupt":
                s+="{}:{} ".format(action,"int")
            elif self.last_status[3]=="conflict":
                s+="{}:({}) ".format(action[0], "|".join(action[1]))

        #s+="{}".format(self._petri)
        

        #for detector, state in self._petri.get_detector_states().items():
        #    s+="{}:{} ".format(detector, state)

        #return "<{}({}) t={}>".format(s,id(self),self.get_time()) 
        #return "<{}({})>".format(s,self.__hash__())
        tokens = self.get_tokens()

        s = "{}\nst:{}\nct:{}\n{}".format(s,"/".join(["{},{}".format(i,tokens[i]) for i in range(len(tokens)) if tokens[i] > 0]),",".join([str(s) for s in self.get_count() if s >0]), hash(self))

        s+="\n"
        for c in self.countdowns:
            s+=".".join([str(i) for i in np.ndarray.flatten(c).tolist() if i>0])
            s+="\n"

        return s
        #tokens = self.get_tokens()
        #return "{}\nst:{}\nct:{}\nt={}\n{}".format(s,"/".join(["{},{}".format(i,tokens[i]) for i in range(len(tokens)) if tokens[i] > 0]),",".join([str(s) for s in self.get_count() if s >0]),",".join(map(lambda s: str(s), self.get_times())), hash(self))

        if len(s)>0:
            outstr = "{}\nt={}".format(s, self.get_time())
        else:
            outstr = "<wait>\nt={}".format(self.get_time())

        return outstr
        #return "<{}@ t={}>".format(self.__hash__(),self.get_time())
        #return "<{}@ t={}>".format(s,self.get_time())
        #return "Time: {}".format(self._time)+self._petri.get_state_repr()+self._petri.get_count_repr()

    def history(self):
        return np.ndarray.flatten(self._petri._n._history[self._petri._n._history>0]).tolist()
