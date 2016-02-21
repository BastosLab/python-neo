# -*- coding: utf-8 -*-
"""
Reading from neuroscope format files.
Ref: http://neuroscope.sourceforge.net/

It is an old format from Buzsaki's lab.

Supported: Read

#TODO:
SpikeTrain file   '.clu'  '.res'
Event  '.ext.evt'  or '.evt.ext'

Author: sgarcia

"""

# needed for python 3 compatibility
from __future__ import absolute_import

import os
from xml.etree import ElementTree

import numpy as np
import quantities as pq

from neo.io.baseio import BaseIO
from neo.io.rawbinarysignalio import RawBinarySignalIO
from neo.core import (Block, Segment,  ChannelIndex,
                      AnalogSignal)


class NeuroScopeIO(BaseIO):
    """


    """

    is_readable = True
    is_writable = False

    supported_objects  = [ Block, Segment , AnalogSignal,  ChannelIndex]

    readable_objects    = [ Block ]
    writeable_objects   = [ ]

    has_header         = False
    is_streameable     = False
    read_params = {
        Segment : [ ]
        }

    # do not supported write so no GUI stuff
    write_params       = None

    name               = 'NeuroScope'

    extensions          = [ 'xml' ]
    mode = 'file'



    def __init__(self , filename = None) :
        """
        Arguments:
            filename : the filename
            
        """
        BaseIO.__init__(self)
        self.filename = filename


        
        


    def read_block(self,
                     lazy = False,
                     cascade = True,
                    ):
        """
        """

        
        tree = ElementTree.parse(self.filename)
        root = tree.getroot()
        acq = root.find('acquisitionSystem')
        nbits = int(acq.find('nBits').text)
        nbchannel = int(acq.find('nChannels').text)
        sampling_rate = float(acq.find('samplingRate').text)*pq.Hz
        voltage_range = float(acq.find('voltageRange').text)
        #offset = int(acq.find('offset').text)
        amplification = float(acq.find('amplification').text)
        
        bl = Block(file_origin = os.path.basename(self.filename).replace('.xml', ''))
        if cascade:
            seg = Segment()
            bl.segments.append(seg)
            
            # RCG
            for i, xml_chx in  enumerate(root.find('anatomicalDescription').find('channelGroups').findall('group')):
                n_channels = len(xml_chx)
                chx = ChannelIndex(name = 'Group {0}'.format(i),
                                            channel_indexes=np.arange(n_channels, dtype = int))
                chx.channel_ids = np.array([int(xml_rc.text) for xml_rc in xml_chx])
                chx.channel_names = np.array(['Channel{0}'.format(id) for id in chx.channel_ids], dtype = 'S')
                bl.channel_indexes.append(chx)
        
            # AnalogSignals
            reader = RawBinarySignalIO(filename = self.filename.replace('.xml', '.dat'))
            seg2 = reader.read_segment(cascade = True, lazy = lazy,
                                                        sampling_rate = sampling_rate,
                                                        t_start = 0.*pq.s,
                                                        unit = pq.V, nbchannel = nbchannel,
                                                        bytesoffset = 0,
                                                        dtype = np.int16 if nbits<=16 else np.int32,
                                                        rangemin = -voltage_range/2.,
                                                        rangemax = voltage_range/2.,)
            for s, sig in enumerate(seg2.analogsignals):
                if not lazy:
                    sig /= amplification
                sig.segment = seg
                seg.analogsignals.append(sig)
                chx.analogsignals.append(sig)
            
        bl.create_many_to_one_relationship()
        return bl

