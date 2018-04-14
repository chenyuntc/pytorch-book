#encoding=utf-8

import math

LOG_ZERO = -99999999.0
LOG_ONE = 0.0

class BeamEntry:
    "information about one single beam at specific time-step"
    def __init__(self):
        self.prTotal=LOG_ZERO       # blank and non-blank
        self.prNonBlank=LOG_ZERO    # non-blank
        self.prBlank=LOG_ZERO       # blank
        self.y=()                   # labelling at current time-step

class BeamState:
    "information about beams at specific time-step"
    def __init__(self):
        self.entries={}

    def norm(self):
        "length-normalise probabilities to avoid penalising long labellings"
        for (k,v) in self.entries.items():
            labellingLen=len(self.entries[k].y)
            self.entries[k].prTotal=self.entries[k].prTotal*(1.0/(labellingLen if labellingLen else 1))

    def sort(self):
        "return beams sorted by probability"
        u=[v for (k,v) in self.entries.items()]
        s=sorted(u, reverse=True, key=lambda x:x.prTotal)
        return [x.y for x in s]

class ctcBeamSearch(object):
    def __init__(self, classes, beam_width, blank_index=0):
        self.classes = classes
        self.beamWidth = beam_width
        self.blank_index = blank_index
        
    def log_add_prob(self, log_x, log_y):
        if log_x <= LOG_ZERO:
            return log_y
        if log_y <= LOG_ZERO:
            return log_x
        if (log_y - log_x) > 0.0:
            log_y, log_x = log_x, log_y
        return log_x + math.log(1 + math.exp(log_y - log_x))
    
    def calcExtPr(self, k, y, t, mat, beamState):
        "probability for extending labelling y to y+k"
        # optical model (RNN)
        if len(y) and y[-1]==k and mat[t-1, self.blank_index] < 0.9:
            return math.log(mat[t, k]) + beamState.entries[y].prBlank
        else:
            return math.log(mat[t, k]) + beamState.entries[y].prTotal
    
    def addLabelling(self, beamState, y):
        "adds labelling if it does not exist yet"
        if y not in beamState.entries:
            beamState.entries[y]=BeamEntry()
    
    def decode(self, inputs, inputs_list):
        """
        Args: 
            inputs(FloatTesnor) :  Output of CTC(batch * timesteps * class)
            inputs_list(list)   :  the frames of each sample
        Returns:
            res(list)           :  Result of beam search
        """
        batches, maxT, maxC = inputs.size()
        res = []
        
        for batch in range(batches):
            mat = inputs[batch].numpy()
            # Initialise beam state
            last=BeamState()
            y=()
            last.entries[y]=BeamEntry()
            last.entries[y].prBlank=LOG_ONE
            last.entries[y].prTotal=LOG_ONE
            
            # go over all time-steps
            for t in range(inputs_list[batch]):
                curr=BeamState()
                
                #跳过概率很接近1的blank帧，增加解码速度
                if (1 - mat[t, self.blank_index]) < 0.1:
                    continue

                #取前beam个最好的结果
                BHat=last.sort()[0:self.beamWidth]
                # go over best labellings
                for y in BHat:
                    prNonBlank=LOG_ZERO
                    # if nonempty labelling
                    if len(y)>0:
                        #相同的y两种可能，加入重复或者加入空白,如果之前没有字符，在NonBlank概率为0
                        prNonBlank=last.entries[y].prNonBlank + math.log(mat[t, y[-1]])     
                            
                    # calc probabilities
                    prBlank = (last.entries[y].prTotal) + math.log(mat[t, self.blank_index])
                    # save result
                    self.addLabelling(curr, y)
                    curr.entries[y].y=y
                    curr.entries[y].prNonBlank = self.log_add_prob(curr.entries[y].prNonBlank, prNonBlank)
                    curr.entries[y].prBlank = self.log_add_prob(curr.entries[y].prBlank, prBlank)
                    prTotal = self.log_add_prob(prBlank, prNonBlank)
                    curr.entries[y].prTotal = self.log_add_prob(curr.entries[y].prTotal, prTotal)
                            
                    #t时刻加入其它的label,此时Blank的概率为0，如果加入的label与最后一个相同，因为不能重复，所以上一个字符一定是blank
                    for k in range(maxC):                                         
                        if k != self.blank_index:
                            newY=y+(k,)
                            prNonBlank=self.calcExtPr(k, y, t, mat, last)
                                    
                            # save result
                            self.addLabelling(curr, newY)
                            curr.entries[newY].y=newY
                            curr.entries[newY].prNonBlank = self.log_add_prob(curr.entries[newY].prNonBlank, prNonBlank)
                            curr.entries[newY].prTotal = self.log_add_prob(curr.entries[newY].prTotal, prNonBlank)
                    
                    # set new beam state
                last=curr
                    
            # normalise probabilities according to labelling length
            last.norm() 
            
            # sort by probability
            bestLabelling=last.sort()[0] # get most probable labelling
            
            # map labels to chars
            res_b =''.join([self.classes[l] for l in bestLabelling])
            res.append(res_b)
        return res

