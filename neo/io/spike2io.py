# -*- coding: utf-8 -*-
"""
exampleio
==================

Classe for fake reading/writing data in a no file.
It is just a example for guidelines for developers who want to develop a new IO.


If you start a new IO class copy/paste and modify.

If you have a problem just mail me or ask the list.


Classes
-------

ExampleIO          - Classe for fake reading/writing data in a no file.


@author : sgarcia

"""


from baseio import BaseIO
from neo.core import *

from numpy import *
from copy import deepcopy

class Spike2IO(BaseIO):
    """
    Class for reading data in smr spike2 CED file.
    
    **Usage**

    **Example**
    
    """
    
    is_readable        = True
    is_writable        = False
    is_object_readable = True
    is_object_writable = False
    has_header         = False
    is_streameable     = False
    read_params        = {
                        Block : [
                                ],
                        Segment : [
                                    ],
                        }
    write_params       = None
    level              = None
    nfiles             = 0
    
    name               = 'Spike 2 CED'
    extensions          = [ 'smr' ]
    objects            = []
    supported_types    = [ Block ]
    
    def __init__(self ) :
        """
        
        **Arguments**
        
        """
        
        BaseIO.__init__(self)


    def read(self , **kargs):
        """
        Read a fake file.
        Return a neo.Block
        See read_block for detail.
        """
        return self.read_block( **kargs)
    
    def read_block(self , filename = '', ):
        """
        Return a fake Block.
        
        **Arguments**
        filename : The filename does not matter.
        
        """
        header = self.read_header(filename = filename)
        print header
        fid = open(filename, 'rb')
        blck = Block()
        
        for i in range(header.channels) :
            channelHeader = header.channelHeaders[i]
            
            if channelHeader.kind !=0:
                print '####'
                print 'channel' , i, 'kind' , channelHeader.kind , channelHeader.type , channelHeader.phy_chan
                print channelHeader
            if channelHeader.kind in [1, 9]:
                print 'analogChanel'
                anaSigs = self.readOneChannelWaveform( fid, i, header ,)
                print len(anaSigs)
                for sig in anaSigs :
                    seg  = Segment()
                    seg._analogsignals = [ sig ]
                    blck._segments.append(seg)
                    
            elif channelHeader.kind in  [2, 3, 4, 5, 8]:
                print 'channel event'
                events = self.readOneChannelEvent( fid, i, header )
                seg  = Segment()
                seg._events +=  events
                blck._segments.append(seg)
            elif channelHeader.kind in  [6, 7]:
                print 'channel spikes'
                spikes, freq = self.readOneChannelEvent( fid, i, header )
                spikeTr = SpikeTrain(spikes = spikes)
                spikeTr.freq = freq
                seg._spiketrains.append(spikeTr)
                
        fid.close()
        
        
        
        return blck
        
        
    def read_header(self , filename = ''):
        
        fid = open(filename, 'rb')
        header = HeaderReader(fid,   dtype(headerDescription))
        
        if header.system_id < 6:
            header.dtime_base = 1e-6
            header.datetime_detail = 0
            header.datetime_year = 0
        
        channelHeaders = [ ]
        for i in range(header.channels):
            # read global channel header
            fid.seek(512 + 140*i) # TODO verifier i ou i-1
            channelHeader = HeaderReader(fid, dtype(channelHeaderDesciption1))
            if channelHeader.kind in [1, 6]:
                dt = [('scale' , 'f4'),
                      ('offset' , 'f4'),
                      ('unit' , 'S6'),]
                channelHeader += HeaderReader(fid, dtype(dt))
                if header.system_id < 6:
                    channelHeader += HeaderReader(fid, dtype([ ('divide' , 'i4')]) )#i8
                else : 
                    channelHeader +=HeaderReader(fid, dtype([ ('interleave' , 'i4')]) )#i8
            if channelHeader.kind in [7, 9]:
                dt = [('min' , 'f4'),
                      ('max' , 'f4'),
                      ('unit' , 'S6'),]
                channelHeader += HeaderReader(fid, dtype(dt))
                if header.system_id < 6:
                    channelHeader += HeaderReader(fid, dtype([ ('divide' , 'i4')]))#i8
                else :
                    channelHeader += HeaderReader(fid, dtype([ ('interleave' , 'i4')]) )#i8
            if channelHeader.kind in [4]:
                dt = [('init_low' , 'u1'),
                      ('next_low' , 'u1'),]
                channelHeader += HeaderReader(fid, dtype(dt))
            
            channelHeader.type = dict_kind[channelHeader.kind]
            channelHeaders.append(channelHeader)
        
        header.channelHeaders = channelHeaders
        
        fid.close()
        return header

            
    def readOneChannelWaveform(self , fid, channel_num, header ,):
        """
        """
        channelHeader = header.channelHeaders[channel_num]
        
        # read AnalogSignal
        
        # data type
        if channelHeader.kind == 1:
            dt = dtype('i2')
        elif channelHeader.kind == 9:
            dt = dtype('f4')
        
        # sample rate
        if header.system_id in [1,2,3,4,5]: # Before version 5
            #print 'calcul freq',channelHeader.divide , header.us_per_time , header.time_per_adc
            sample_interval = (channelHeader.divide*header.us_per_time*header.time_per_adc)*1e-6
        else :
            sample_interval = (channelHeader.l_chan_dvd*header.us_per_time*header.dtime_base)
        #print 'sample_interval' , sample_interval
        freq = 1./sample_interval
        #print 'freq' , freq
        
        # read blocks header
        fid.seek(channelHeader.firstblock)
        anaSig = AnalogSignal()
        anaSig.signal = array([ ] , dtype = 'f')
        anaSig.freq = freq
        anaSigs = [ ]
        for b in range(channelHeader.blocks) :
            blockHeader = HeaderReader(fid, dtype(blockHeaderDesciption))
            
            # read data
            sig = fromstring( fid.read(blockHeader.items*dt.itemsize) , dtype = dt)
            
            # convert for int16
            if dtype.kind == 'i' :
                sig = sig.astype('f4') *channelHeader.scale/ 6553.6 + channelHeader.offset
            
            # add to prev block
            anaSig.signal = concatenate( ( anaSig.signal , sig ))
            
            # jump to next block
            if blockHeader.succ_block > 0 :
                fid.seek(blockHeader.succ_block)
                nextBlockHeader = HeaderReader(fid, dtype(blockHeaderDesciption))
                
                # check is there a continuity with next block
                sample_interval = (blockHeader.end_time-blockHeader.start_time)/(blockHeader.items-1)
                #print 'sample_interval' , sample_interval, sample_interval*header.us_per_time*header.time_per_adc*1e-6
                interval_with_next = nextBlockHeader.start_time - blockHeader.end_time
                if interval_with_next > sample_interval:
                    # discontinuous :
                    # create a new anaSig
                    #print 'rupture' , sample_interval , interval_with_next
                    anaSigs.append(anaSig)
                    anaSig = AnalogSignal()
                    anaSig.signal = array([ ] , dtype = 'f')
                    anaSig.freq = freq
                
                fid.seek(blockHeader.succ_block)
        # last one
        anaSigs.append(anaSig)
        
        # TODO gerer heure et freq verifier
        return anaSigs
        
    def readOneChannelEvent(self , fid, channel_num, header ,):
        channelHeader = header.channelHeaders[channel_num]
        
        alltrigs = None
        if channelHeader.kind in [2, 3, 4 , 5 , 6 ,7, 8]:
            if channelHeader.firstblock >0 :
                fid.seek(channelHeader.firstblock)
            for b in range(channelHeader.blocks) :
                print '  block' , b 
                blockHeader = HeaderReader(fid, dtype(blockHeaderDesciption))
                print  '  items in block' , blockHeader.items
                
                # common for kind 5 6 7 8 9
                format5 = [('tick' , 'i4') , ('marker' , 'i4') ]
#                               ('markers0' , 'u1'),
#                               ('markers1' , 'u1'),
#                               ('markers2' , 'u1'),
#                               ('markers3' , 'u1')]
                
                if channelHeader.kind in [2, 3, 4]:
                    # Event data
                    format = [('tick' , 'i4') ]
                elif channelHeader.kind in [5]:
                    # Marker data
                    format = format5
                elif channelHeader.kind in [6]:
                    # AdcMark data
                    n_extra = channelHeader.n_extra/2 # 2 bytes
                    format = deepcopy(format5)
                    for n in range(n_extra) :
                        format += [ ('adc%d'%n , 'i2')]
                elif channelHeader.kind in [7]:
                    #  RealMark data
                    n_extra = channelHeader.n_extra/4 # 4 bytes
                    format = deepcopy(format5)
                    for n in range(n_extra) :
                        format += [ ('real%d'%n , 'f4')]
                elif channelHeader.kind in [8]:
                    # TextMark data
                    n_extra = channelHeader.n_extra # 1 bytes
                    format = deepcopy(format5)
                    format += [ ('label' , 'S%d'%n_extra)]
                
                # read all events in block
                dt = dtype(format)
                trigs = fromstring( fid.read( blockHeader.items*dt.itemsize)  , dtype = dt)
                
                if alltrigs is None :
                    alltrigs = trigs
                else :
                    alltrigs = concatenate( (alltrigs , trigs))
                
                        
                    
                if blockHeader.succ_block > 0 :
                    fid.seek(blockHeader.succ_block)
                # TODO verifier time
        
        if alltrigs is None : return [ ]
        
        #  convert in neo standart class : event or spiketrains
        alltimes = alltrigs['tick'].astype('f')*header.us_per_time * header.dtime_base
        
        if channelHeader.kind in [2, 3, 4 , 5 ,  8]:
            # event
            events = [ ]
            for t,time in enumerate(alltimes) :
                event = Event(time = time)
                if channelHeader.kind >= 5:
                    #print '        trig: ', alltrigs[t]
                    event.marker = alltrigs[t]['marker'] # TODO 4 marker u1 ou 1 marker i4
                if channelHeader.kind == 8:
                    print 'label' , alltrigs[t]['label']
                    event.marker = alltrigs[t]['label']
                events.append(event)
                
            return events
            
        elif channelHeader.kind in [6 , 7]:
            # sample rate
            if header.system_id in [1,2,3,4,5]:
                print 'calcul freq',channelHeader.divide , header.us_per_time , header.time_per_adc
                sample_interval = (channelHeader.divide*header.us_per_time*header.time_per_adc)*1e-6
            else :
                sample_interval = (channelHeader.l_chan_dvd*header.us_per_time*header.dtime_base)
            freq = 1./sample_interval
            print 'freq' , freq
        
            spikes =  [ ]
            for t,time in enumerate(alltimes) :
                spike = Spike()
                spike.time = time
                #print alltrigs[t]
                waveform = array(list(alltrigs[t])[2:])
                if channelHeader.kind == 6 :
                    waveform = waveform.astype('f4') *channelHeader.scale/ 6553.6 + channelHeader.offset
                spike.waveform = waveform
                spike.marker = alltrigs[t]['marker']
                spikes.append(spike)
            return spikes, freq






class HeaderReader(object):
    def __init__(self , fid , dtype):
        if fid is not None :
            array = fromstring( fid.read(dtype.itemsize) , dtype)[0]
        else :
            array = zeros( (1) , dtype = dtype)[0]
        object.__setattr__(self, 'dtype' , dtype)
        object.__setattr__(self, 'array' , array)
        
    def __setattr__(self, name , val):
        if name in self.dtype.names :
            self.array[name] = val
        else :
            object.__setattr__(self, name , val)

    def __getattr__(self , name):
        if name in self.dtype.names :
            return self.array[name]
        else :
            object.__getattr__(self, name )
    def names(self):
        return self.array.dtype.names
    
    def __repr__(self):
        print 'HEADER'
        for name in self.dtype.names :
            if self.dtype[name].kind != 'S' :
                print name , self.array[name]
        print ''
        return ''
    
    def __add__(self, header2):
#        print 'add' , self.dtype, header2.dtype
        newdtype = [ ]
        for name in self.dtype.names :
            newdtype.append( (name , self.dtype[name].str) )
        for name in header2.dtype.names :
            newdtype.append( (name , header2.dtype[name].str) )
        newdtype = dtype(newdtype)
        newHeader = HeaderReader(None , newdtype )
        newHeader.array = fromstring( self.array.tostring()+header2.array.tostring() , newdtype)[0]
        return newHeader

# headers structures :
headerDescription = [
    ( 'system_id', 'i2' ),
    ( 'copyright', 'S10' ),
    ( 'creator', 'S8' ),
    ( 'us_per_time', 'i2' ),
    ( 'time_per_adc', 'i2' ),
    ( 'filestate', 'i2' ),
    ( 'first_data', 'i4' ),#i8
    ( 'channels', 'i2' ),
    ( 'chan_size', 'i2' ),
    ( 'extra_data', 'i2' ),
    ( 'buffersize', 'i2' ),
    ( 'os_format', 'i2' ),
    ( 'max_ftime', 'i4' ),#i8
    ( 'dtime_base', 'f8' ),
    ( 'datetime_detail', 'u1' ),
    ( 'datetime_year', 'i2' ),
    ( 'pad', 'S52' ),
    ( 'comment1', 'S80' ),
    ( 'comment2', 'S80' ),
    ( 'comment3', 'S80' ),
    ( 'comment4', 'S80' ),
    ( 'comment5', 'S80' ),
    ]

channelHeaderDesciption1 = [
    ('del_size','i2'),
    ('next_del_block','i4'),#i8
    ('firstblock','i4'),#i8
    ('lastblock','i4'),#i8
    ('blocks','i2'),
    ('n_extra','i2'),
    ('pre_trig','i2'),
    ('free0','i2'),
    ('py_sz','i2'),
    ('max_data','i2'),
    ('comment','S72'),
    ('max_chan_time','i4'),#i8
    ('l_chan_dvd','i4'),#i8
    ('phy_chan','i2'),
    ('title','S10'),
    ('ideal_rate','f4'),
    ('kind','u1'),
    ('unused1','i1'),
    
    ]

dict_kind = {
    0 : 'empty',
    1: 'Adc',
    2: 'EventFall',
    3: 'EventRise',
    4: 'EventBoth',
    5: 'Marker',
    6: 'AdcMark',
    7: 'RealMark',
    8: 'TextMark',
    9: 'RealWave',
    }


blockHeaderDesciption =[
    ('pred_block','i4'),#i8
    ('succ_block','i4'),#i8
    ('start_time','i4'),#i8
    ('end_time','i4'),#i8
    ('channel_num','i2'),
    ('items','i2'),
    ]
