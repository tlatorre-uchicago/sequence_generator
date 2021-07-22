import numpy as np

from fractions import Fraction
import matplotlib.pyplot as plt
import math
import csv
import random
import yaml
import json
from collections import deque
import math
import copy
from bitstring import BitArray

from phd import viz
from datetime import datetime
import os
import os, shutil

from bokeh.io import output_file, show
from bokeh.layouts import gridplot
from bokeh.plotting import figure
from bokeh.layouts import column
from bokeh.models import Range1d
from bokeh.models import Span, BoxAnnotation, ColumnDataSource
import phd.viz as viz
Colors, pallet = viz.bokeh_theme(return_color_list=True)

from pathlib import Path
from os.path import join

def save_multi_csv(pulse_sequences, channels, filename):
    """
    Saves multiple pulse sequences in a single CSV file and writes header
    information so that the Keysight AWG knows which channels correspond to
    which columns. Also writes the sample rate in the header.
    """
    sample_times = [seq.sampleTime for seq in pulse_sequences]

    if len(set(sample_times)) > 1:
        raise ValueError("All pulse sequences must have the same sample time!")

    sample_rate = 1.0/sample_times[0]

    # Use pathlib.Path since we are using Python 3 and it makes filenames much
    # easier to work with on Windows.
    with open(Path(filename), 'w', newline='') as f:
        if sample_rate > 1e9:
            f.write("SampleRate = %.3f GHz\r\n" % (sample_rate/1e9))
        else:
            f.write("SampleRate = %.3f MHz\r\n" % (sample_rate/1e6))
        f.write(','.join(["Y%i" % channel for channel in channels]) + '\r\n')
        writer = csv.writer(f)
        writer.writerows(zip(*[seq.sequence for seq in pulse_sequences]))

def lcm(x, y):
   # choose the greater number
   if x > y:
       greater = x
   else:
       greater = y
   while(True):
       if((greater % x == 0) and (greater % y == 0)):
           lcm = greater
           break
       greater += 1
   return lcm

def determine_ppm_properties(params):
    """
    The sequence length must be chosen carefully for several reasons. First, it must be a multiple of 128
    due to hardware limitations of the M8196 AWG. Second, it must be a multiple of some number of samples
    that take the same amount of time to run as some other integer number of laser pulses. Often this isn't
    complicated, say if the laser rep rate is 20GHz and the AWG runs at 80Gsps. Then 4 samples pass for each
    laser pulse, and the sequence must therefore be a multiple of 4 samples. (It already is from the 128
    sample requirment). Managing this requirement is more important when the AWG sample rate is not a
    nice multiple of the laser rep rate.
    """

    sample_rate = params["system"]["sync_rate"]*32
    # say, for example, laser rep rate is 20GHz and divider is 7
    # ((20GHz/7)*32)/20 is just 32/7 so pulse discretization is 7
    # sample discretization is the numerator of 32/7, so it's always 32
    pulses_disc = params["system"]["divider"]
    samples_disc = 32

    print("samples_disc", samples_disc)
    print("pulses_disc", pulses_disc)
    # there must be integer number of pulses in each PPM cycle.
    m_value = params["ppm"]["m_value"]
    minimum_dead_time = params["ppm"]["minimum_dead_time"]
    laser_time = 1/(params["system"]["laser_rate"]*1e9) # laser period in seconds
    print(laser_time)

    # how long is each PPM cycle? If the time for M pulses plus the minimum
    # dead time is not an integer number of laser pulses, the dead time is increased
    # until the cycle is an integer number of laser pulses.
    # note: 1 cycle does not have to be an integer number of AWG samples. That
    # is handled at the sequence level. (of course it has to be an integer
    # number of laser pulses)
    minimum_dead_pulses = math.ceil((minimum_dead_time/1e9)/laser_time)
    # print("minimum dead pulses: ", minimum_dead_pulses)

    pulses_per_cycle = m_value + minimum_dead_pulses
    # print("pulses_per_cycle: ", pulses_per_cycle)
    # the number of pulses that equates to an integer number of samples in time is a 'set'
    # a cycle is a PPM symbol with dead time


    # what is maximum number of laser pulses AWG could modulate?
    # final sequence must be multiple of 128 (hardware limitation)
    # sample_set is minimum number of samples that is valid for awg and lasts integer number of laser pulses
    # pulse_set is (integer) number of pulses that pass in same time as sample_set
    sample_set = lcm(128,samples_disc)
    pulse_set = (sample_set/samples_disc)*pulses_disc #SHOULD be integer
    ### print("SAMPLE SET: ", sample_set)
    ### print("PULSE SET: ", pulse_set)
    # maximum valid sequence length that is integer number of laser pulses
    max_valid_samples = (((params["awg"]["max_sample_depth"]*1000)//sample_set)*sample_set)
    # print("max valid samples:", max_valid_samples)
    max_valid_pulses = (max_valid_samples/samples_disc)*pulses_disc #SHOULD be integer
    # print("max valid pulses: ", max_valid_pulses)

    #OUTPUT
    cycles = math.floor(max_valid_pulses/pulses_per_cycle)
    # pulses_per_cycle doesn't have to be a multiple of pulse_set. But cycles*pulses_per_cycle should be
    # print("cycles: ", cycles)
    # data_pulses is number of pulses that are either part of PPM word or deadtime.
    data_pulses = cycles*pulses_per_cycle
    # if data_pulses/pulse_set was not an integer, add some extra samples at the end until it is

    if params["ppm"]["extension_mode"] == "OP1":
        total_sets = math.ceil(data_pulses/pulse_set)
        print("Using extension mode 1. Total sets: ", total_sets)
    elif params["ppm"]["extension_mode"] == "OP2":
        total_sets = math.ceil(max_valid_pulses/pulse_set)
        print("Using extension mode 2. Total sets: ", total_sets)
    else:
        print("ppm.extension_mode key error")
        return 1

    # OUTPUT
    total_pulses = pulse_set*total_sets
    print("total pulses: ", total_pulses)
    # OUTPUT
    total_samples = sample_set*total_sets
    print("total samples: ", total_samples)

    params["cycles_per_sequence"] = cycles
    params["total_pulses"] = int(total_pulses)
    params["total_samples"] = int(total_samples)
    params["pulses_per_cycle"] = pulses_per_cycle
    params["sample_rate"] = sample_rate # in GSamples/s
    params["laser_time"] = laser_time # in seconds
    params["sample_time"] = 1/(sample_rate*1e9) # in seconds
    return params


def determine_regular_properties(params, cycles = None):
    """
    The main difference between this and determine_ppm_properties is that regular sequences are
    not extended. Here, the attempt is made to pick a sequence length that repeats perfectly so
    all dead-times are the same, or the modulated 'on' pulse-rate is constant. There isn't a pulse at the
    end of the sequence that has no dead-time, or a dead-time that is too long.
    """
    if params["system"]["sample_rate_override"] == -1:
        sample_rate = params["system"]["sync_rate"]*32
        # say, for example, laser rep rate is 20GHz and divider is 7
        # ((20GHz/7)*32)/20 is just 32/7 so pulse discretization is 7
        # sample discretization is the numerator of 32/7, so it's always 32
        pulses_disc = params["system"]["divider"]
        samples_disc = 32

        print("samples_disc", samples_disc)
        print("pulses_disc", pulses_disc)
    else:
        # override mode
        sample_rate = params["system"]["sample_rate_override"]
        samples_disc = int(sample_rate/params["system"]["laser_rate"])
        pulses_disc = 1


    print("sample_rate", sample_rate)
    laser_time = 1/(params["system"]["laser_rate"]*1e9) # laser period in seconds
    print(laser_time)

    # how long is each PPM cycle? If the time for M pulses plus the minimum
    # dead time is not an integer number of laser pulses, the dead time is increased
    # until the cycle is an integer number of laser pulses.
    # note: 1 cycle does not have to be an integer number of AWG samples. That
    # is handled at the sequence level. (of course it has to be an integer
    # number of laser pulses)
    # minimum_dead_pulses = math.ceil((minimum_dead_time/1e9)/laser_time)
    # print("minimum dead pulses: ", minimum_dead_pulses)

    pulses_per_cycle = params["regular"]["data"]["pulse_divider"]

    # print("pulses_per_cycle: ", pulses_per_cycle)
    # the number of pulses that equates to an integer number of samples in time is a 'set'
    # a cycle is a PPM symbol with dead time


    # what is maximum number of laser pulses AWG could modulate?
    # final sequence must be multiple of 128 (hardware limitation)
    # sample_set is minimum number of samples that is valid for awg and lasts integer number of laser pulses
    # pulse_set is (integer) number of pulses that pass in same time as sample_set
    sample_set = lcm(128,samples_disc)
    pulse_set = (sample_set/samples_disc)*pulses_disc #SHOULD be integer

    pulse_set_regular = lcm(pulse_set,pulses_per_cycle)  # there can be multiple cycles per regular set
    sample_set_regular = (pulse_set_regular/pulse_set)*sample_set # should be an integer
    print("pulse_set_regular: ", pulse_set_regular)
    print("sample_set_regular: ", sample_set_regular)

    ### print("SAMPLE SET: ", sample_set)
    ### print("PULSE SET: ", pulse_set)
    # maximum valid sequence length that is integer number of laser pulses
    max_valid_samples = (((params["awg"]["max_sample_depth"]*1000)//sample_set_regular)*sample_set_regular)
    print("max_valid_samples: ", max_valid_samples)
    # print("max valid samples:", max_valid_samples)
    if max_valid_samples == 0:
        print("A nonzero length regular sequence could not be found. Choose a different "
              "clock divider value or a smaller pulse divder.")
        exit(1)
    max_valid_pulses = (max_valid_samples/samples_disc)*pulses_disc #SHOULD be integer
    # print("max valid pulses: ", max_valid_pulses)

    #OUTPUT
    if cycles is None:
        cycles = max_valid_pulses/pulses_per_cycle # should be an integer
        print("cycles: ", cycles)

    # pulses_per_cycle doesn't have to be a multiple of pulse_set. But cycles*pulses_per_cycle should be
    # data_pulses is number of pulses that are either part of PPM word or deadtime.



    # OUTPUT
    total_pulses = cycles*pulses_per_cycle
    print("total pulses: ", total_pulses)
    # OUTPUT
    total_samples = (total_pulses/pulse_set)*sample_set  # should be in integer
    print("total samples: ", total_samples)

    params["cycles_per_sequence"] = int(cycles)
    params["total_pulses"] = total_pulses
    params["total_samples"] = int(total_samples)
    params["pulses_per_cycle"] = pulses_per_cycle
    params["sample_rate"] = sample_rate # in GSamples/s
    params["laser_time"] = laser_time # in seconds
    params["sample_time"] = 1/(sample_rate*1e9) # in seconds
    return params


def gaussian(sigma, center, time):
    # sigma = pulse_width / (2 * np.sqrt(2 * np.log(2)))
    return np.exp(-(time - center) ** 2 / (2 * sigma ** 2))


#need: sampleTime, laserTime in pulse class

# need sequence length, cycleTime, laserTime in sequence class

# in the future I might want to have different hightimes for pulses that are in the center of samples. Or vice versa.
# then every pulse should have its own object. So that a special method could be written that handles those cases.
class pulse():
    def __init__(self, hightime, risetime, center, sampleTime, amplitude, laserTime):
        self.hightime = hightime*laserTime
        self.risetime = risetime*laserTime
        self.center = center
        self.sampleTime = sampleTime
        self.amplitude = amplitude
        self.left_center = 0
        self.right_center = 0
        self.nearCenterSampleIndex = 0
        self.laserTime = laserTime
        self.nearCenterSampleIndex = self.center // self.sampleTime
        self.index_left = int(self.nearCenterSampleIndex)
        self.index_right = int(self.nearCenterSampleIndex + 1)

    def writePulseGaussian(self, time):
        # defines the theoretical (smooth, unsampled) pulse with risetime, hightime, and center.
        # writePulseFragment() calls this
        if (self.center - 0.5 * self.hightime) < time < (self.center + 0.5 * self.hightime):
            amp = self.amplitude
        elif time <= (self.center - 0.5 * self.hightime):
            self.left_center = round((self.center - 0.5 * self.hightime) / self.sampleTime) * self.sampleTime
            amp = self.amplitude * gaussian(self.risetime, self.left_center, time)
        elif time >= (self.center + 0.5 * self.hightime):
            self.right_center = round((self.center + 0.5 * self.hightime) / self.sampleTime) * self.sampleTime

            amp = self.amplitude * gaussian(self.risetime, self.right_center, time)
        else:
            # print("error")
            print(time)
            print(self.hightime)
            exit(1)
        return amp

    def writeSampledPulse(self, sequence_):
        #Writes several samples into the main sequence that form a pulse

        minLeftAmplitude = 1
        minRightAmplitude = 1
        req = 0


        # write the pulse to the list iterating out from near the center
        while minLeftAmplitude > .002 or minRightAmplitude > .002:
            # print("location of pulse between indexes:", pulse_time/sampleTime)
            sequence_[self.index_left] = self.writePulseGaussian(self.index_left * self.sampleTime)
            if sequence_[self.index_left] <= minLeftAmplitude:
                minLeftAmplitude = sequence_[self.index_left]
            self.index_left = self.index_left - 1


            sequence_[self.index_right] = self.writePulseGaussian(self.index_right * self.sampleTime)
            if sequence_[self.index_right] <= minRightAmplitude:
                minRightAmplitude = sequence_[self.index_right]
            self.index_right = self.index_right + 1
            req = req + 1

    def plotSampledPulse(self, sequence_, count_, slot_, figcount):
        x = []
        y = []
        for i in range(self.index_left - int(2 * self.laserTime // self.sampleTime),
                       self.index_right + int(2 * self.laserTime // self.sampleTime)): x.append(i * self.sampleTime * 1e9)

        for i in range(self.index_left - int(2 * self.laserTime // self.sampleTime),
                       self.index_right + int(2 * self.laserTime // self.sampleTime)): y.append(sequence_[i])

        plt.figure(figcount)
        plt.axvline(self.left_center * 1e9, color="#d1d1d1")
        plt.axvline(self.right_center * 1e9, color="#d1d1d1")

        plt.plot(x, y, color="#4189c4")

        plt.axvline(self.center * 1e9, color="#ffc4b5")
        plt.axvline((self.center + self.laserTime) * 1e9, color="#ffc4b5")
        plt.axvline((self.center - self.laserTime) * 1e9, color="#ffc4b5")
        plt.axvline((self.center + 2 * self.laserTime) * 1e9, color="#ffc4b5")
        plt.axvline((self.center - 2 * self.laserTime) * 1e9, color="#ffc4b5")
        # print("pulse laser time: ", pulse_time*1e9)
        # print("pulse laser time: ", pulse_time*1e9)
        # print("pulse laser time plus 1: ", pulse_time*1e9)

        plt.xlabel("time (ns)")
        plt.ylabel("Amplitude")
        plt.title("pulse in cycle number " + str(count_ + 1) +
                  " in slot " + str(slot_))
        plt.show()

class PulseSequence(object):
    def __init__(self, params, time_data = None,amp_data = None, clock_sequence = False):
        self.params = params
        if time_data is None and amp_data is not None:
            # regular mode
            time_data = np.zeros(len(amp_data))
        elif time_data is not None and amp_data is None:
            # ppm mode
            amp_data = np.ones(len(time_data))
        elif clock_sequence is True:
            self.sequence = np.zeros(params["total_samples"])
            self.sequence[0:params["total_samples"]//10] = 1
            self.sampleTime = params["sample_time"]
            return # clock sequences don't need further set up
        else:
            print("expected either time_data or amp_data or a True val for clock_sequence. Exiting. ")
            exit(1)

        self.pulseList = []  # this will be a list of pulse objects.
        self.times = np.zeros(len(time_data))
        self.timesSequence = time_data
        self.ampSequence = amp_data
        self.padding = params["ppm"]["padding"]
        self.sampleTime = params["sample_time"]
        self.sequence = np.zeros(params["total_samples"])
        self.cycleTime = params["pulses_per_cycle"]*params["laser_time"]
        self.laserTime = params["laser_time"]
        self.figcount = 1

        if self.params["system"]["mode"] == "ppm":
            self.symbol_start = np.zeros(len(time_data))
            self.symbol_end = np.zeros(len(time_data))

    def generate_times_list(self):
        # generate at list of universal AWG pulse times from the start time of each cycle
        # these AWG pulse times align with pulses of the laser
        self.cycleStart = self.padding
        for i in range(len(self.times)):
            self.times[i] = self.cycleStart + self.timesSequence[i] * self.params["laser_time"]

            # for making the boxes in the preview images
            if self.params["system"]["mode"] == "ppm":
                self.symbol_start[i] = self.cycleStart
                self.symbol_end[i] = self.cycleStart + self.params["ppm"]["m_value"]*self.params["laser_time"]

            # the actual PPM symbol location
            self.cycleStart = self.cycleStart + self.cycleTime



    def gen_pulse_objects(self):
        # this writes a list of default pulse objects
        for i in range(len(self.timesSequence)):
            amplitude = self.params["awg"]["amplitude_multiplier"]*self.ampSequence[i]
            self.pulseList.append(pulse(self.params["pulse"]["high_time"], self.params["pulse"]["sigma"], self.times[i],
                                        self.sampleTime, amplitude, self.params["laser_time"]))
        # print(self.pulseList)

    # def apply_amplitude_array(self,array):
    #     # length of array and pulseList should be the same
    #     for amp, pulseObject in zip(array,self.pulseList):
    #         pulseObject.amplitude = amp

    def write_pulses(self):
        # Here I loop over a list of pulse objects and 'write' each one'
        # each pulse object already knows where and how to write itself at this point
        #
        # pulseObject should know itself based on data if it should write a pulse or not...
        for i, pulseObject in enumerate(self.pulseList):
            pulseObject.writeSampledPulse(self.sequence)

    def plot_some_pulses(self, start, end):
        for count, pulseObject in enumerate(self.pulseList[start:end]):
            pulseObject.plotSampledPulse(self.sequence1, count, self.timesSequence[count],self.figcount)
            self.figcount = self.figcount + 1

    def delay_channel(self, delay, delay_info=False):
        delaySequence = deque(self.sequence)
        delaySamples = int((delay / 1e12) / self.sampleTime)
        delay_s = delay/1e12
        delaySequence.rotate(delaySamples)
        self.times = self.times + delay_s # for plotting
        self.sequence = list(delaySequence)
        if self.params["system"]["mode"] == "ppm":
            self.symbol_start = self.symbol_start + delay_s
            self.symbol_end = self.symbol_end + delay_s
            if delay_info:
                print("Channel delayed by ", delaySamples * round(self.sampleTime * 1e12, 2), " picoseconds, or ",
                      delaySamples, "samples.")




    def plot_sequence_portion(self, start, end=None, linking_axis = None, title = None):
        if end is None:
            end = len(self.sequence)
        else:
            if end > len(self.sequence):
                end = len(self.sequence)
            else:
                end = end

        # the main data ##############
        source = ColumnDataSource(data=dict(
            x=np.arange(start, end) * self.sampleTime * 1e9,
            y=np.array(self.sequence[start:end])
        ))
        TOOLTIPS = [
            ("index", "$index"),
            ("(x,y)", "($x, $y)")
        ]
        Tools = "pan,wheel_zoom,box_zoom,reset,xwheel_zoom"

        s1 = figure(plot_width=1600,
                    plot_height=600,
                    x_range=linking_axis,  # should be ignored if None type
                    title=title,
                    output_backend="webgl",
                    tools=Tools,
                    active_scroll='xwheel_zoom', tooltips=TOOLTIPS)

        # added features for the plot ##############
        spans = []
        laserTimes = []
        if self.params["system"]["mode"] == "ppm":
            for st, en in zip(self.symbol_start, self.symbol_end):
                box = BoxAnnotation(left=st*1e9, right=en*1e9, fill_alpha=0.1, fill_color='green')
                s1.add_layout(box)

            timesRange = (np.arange(1, 20) - 10) * self.laserTime * 1e9
            for time in self.times:
                pulseTimes = time * 1e9 + timesRange
                for pulseTime in pulseTimes:
                    laserTimes.append(pulseTime)


        if self.params["system"]["mode"] == "regular":
            timesRange = (np.arange(0, self.params["pulses_per_cycle"])) * self.laserTime * 1e9
            for time in self.times:
                pulseTimes = time * 1e9 + timesRange
                for pulseTime in pulseTimes:
                    laserTimes.append(pulseTime)



        if end <= 10000 or self.params["system"]["mode"] == "ppm":
            # don't add the laser lines the plot length is really long
            plotLines = []
            for time in laserTimes:
                plotLines.append(Span(location=time, dimension='height', line_color='red', line_width=1.8, line_alpha = 0.1))
            s1.renderers.extend(plotLines)

        s1.line('x', 'y', source=source)
        s1.xaxis.axis_label = "time (ns)"
        s1.yaxis.axis_label = "Amplitude"
        return s1

    def save(self, path, Chname, sequence_number = 0, precursor = ''):
        # Change this if you generalize this script later
        today_now = datetime.now().strftime("%y.%m.%d.%H.%M")
        if precursor != '':
            precursor = precursor + '_'
        fname = f'{precursor}{sequence_number}_{today_now}_{Chname}'
        full_path = os.path.join(path,fname + '.csv')
        # I think you need the zero sequence for imaginary data?
        self.zerosequence = [0] * self.params["total_samples"]
        with open(full_path , 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(zip(self.sequence, self.zerosequence))

    # def saveSimple(self, path, Chname, time, sequence_number = 0, precursor = ''):
    #     #Change this if you generalize this script later
    #     if Ch == 1:
    #         seq = self.sequence1
    #     if Ch == 2:
    #         seq = self.sequence2
    #     if Ch == 3:
    #         seq = self.clock_sequence
    #
    #
    #
    #     if precursor != '':
    #         precursor = precursor + '_'
    #     fname = f'{precursor}{sequence_number}_{time}_{Chname}'
    #     full_path = os.path.join(path,fname + '.csv')
    #
    #     with open(full_path , 'w', newline='') as f:
    #         writer = csv.writer(f)
    #         writer.writerow(zip(self.sequence1, self.sequence2, self.clock_sequence))
    #         #writer.writerows(zip(seq, self.zerosequence))

    def save_bin8(self, path, Chname, time, sequence_number = -1, precursor = '', save_file_params = False):

        if precursor != '':
            precursor = precursor + '_'
        fname = f'{precursor}{sequence_number}_{time}_{Chname}'
        fname_bin = fname + '.bin'
        fname_yml = fname + '.yml'

        full_path_bin = os.path.join(path,fname_bin)
        full_path_yml = os.path.join(path, fname_yml)
        array = np.array(self.sequence)
        array = array*127 #will be discretized as int8 (-128 to +127)
        # the < marker specifies little endian
        # array.astype("<i2").tofile(full_path)
        array.astype("int8").tofile(full_path_bin)

        if sequence_number == 0:
            self.params["fileName1"] = fname # this is used for loading the dataset into the awg

        if save_file_params:
            file_params = {"times": self.times.tolist(), "times_sequence": self.timesSequence}
            with open(full_path_yml,'w') as f:
                yaml.dump(file_params, f)

    def save_csv(self, path, Chname, time, sequence_number = -1, precursor = '', save_file_params = False):

        if precursor != '':
            precursor = precursor + '_'
        fname = f'{precursor}{sequence_number}_{time}_{Chname}'
        fname_csv = fname + '.csv'
        fname_yml = fname + '.yml'

        full_path_csv= os.path.join(path,fname_csv)
        full_path_yml = os.path.join(path, fname_yml)
        array = np.array(self.sequence)
        #array = array*127 #will be discretized as int8 (-128 to +127)
        # the < marker specifies little endian
        # array.astype("<i2").tofile(full_path)
        #array.astype("int8").tofile(full_path_bin)

        self.zero_sequence = [0]*len(self.sequence)

        with open(full_path_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(zip(self.sequence, self.zero_sequence))

        if sequence_number == 0:
            self.params["fileName1"] = fname # this is used for loading the dataset into the awg

        if save_file_params:
            file_params = {"times": self.times.tolist(), "times_sequence": self.timesSequence}
            with open(full_path_yml,'w') as f:
                yaml.dump(file_params, f)



    def saveParams(self, sequence_number):

        if sequence_number == 0:
            print("Saving Params File")
            yaml_name = '0_' + datetime.now().strftime("%y.%m.%d.%H.%M") + '_params.yml'

            yaml_path = os.path.join(self.params['Output']['save_path'],yaml_name)
            with open(yaml_path, 'w') as f:
                data = yaml.dump(self.params, f)
        else:
            pass



def generate_qkd_amplitudes(data1, data2, amp1, amp2, amp_phase):
    Amp1 = np.zeros(len(data1))
    Amp2 = np.zeros(len(data2))
    for i, (d1,d2) in enumerate(zip(data1,data2)):
        if d1 == 0 and d2 == 1:
            Amp1[i] = 0
            Amp2[i] = amp2
        if d1 == 1 and d2 == 0:
            Amp1[i] = amp1
            Amp2[i] = 0
        if d1 == 1 and d2 == 1:
            Amp1[i] = amp_phase
            Amp2[i] = amp_phase

    return Amp1, Amp2

def manage_ppm(ppm_dict, data):
    # split the data into chunks that fit in awg memory
    chunk_size = ppm_dict["cycles_per_sequence"]
    #print(data)
    sequences = math.ceil(len(data)/chunk_size)
    print("There are ", sequences, " Sequences, with each containing ", chunk_size, "symbols")
    j = 0
    today_now = datetime.now().strftime("%y.%m.%d.%H.%M")
    for i in range(sequences): #loop through the data
    #for i in range(1):  # loop through the data
        current_data = data[j:j + chunk_size]
        # print("length of current data: ", len(current_data))
        j = j + chunk_size
        Delay = ppm_dict["ppm"]["2nd_channel_delay"]
        # print("Delay is: ", Delay)
        channel_1 = PulseSequence(ppm_dict, time_data = current_data)
        channel_1.generate_times_list()
        channel_1.gen_pulse_objects()
        channel_1.write_pulses()

        channel_2 = copy.deepcopy(channel_1)

        channel_1.delay_channel(1000)
        channel_2.delay_channel(1000)
        if i == 1:
            channel_2.delay_channel(Delay, delay_info=True)
        else:
            channel_2.delay_channel(Delay)
    #         # if i == 3:
        #     #PulseSeq.plot_some_pulses(0,12)
        #     PulseSeq.plotSequencePortion(0,-1)
        if ppm_dict["Output"]["clock_gen"]:
            clock = PulseSequence(ppm_dict, clock_sequence = True)

        if ppm_dict["Output"]["file_save"]:
            path = ppm_dict["Output"]["save_path"]
            # PulseSeq.saveSimple(path, 1, 'CH1', today_now, sequence_number = i)
            channel_1.save_bin8(path, 'CH1', today_now, sequence_number=i,save_file_params=True)
            channel_2.save_bin8(path, 'CH2', today_now, sequence_number=i)
            clock.save_bin8(path, 'CLK', today_now, sequence_number=i)
            channel_1.saveParams(i)  # just save params for the first file
    s1 = channel_1.plot_sequence_portion(0,50000)
    s2 = channel_2.plot_sequence_portion(0, 50000, linking_axis=s1.x_range)
    output_file("graphs.html")
    show(column(s1,s2))




def manage_regular(reg_dict, reg_data1, reg_data2):
    """
    Assumes 2 channels of regular data for now
    """
    chunk_size = reg_dict["cycles_per_sequence"]
    # print(data)
    sequences = math.ceil(len(reg_data1) / chunk_size)
    print("There are ", sequences, " Sequences, with each containing ", chunk_size, "symbols")
    j = 0
    today_now = datetime.now().strftime("%y.%m.%d.%H.%M")
    for i in range(sequences):  # loop through the data
        print(j)
        print(chunk_size)
        current_data1 = reg_data1[j:j + chunk_size]
        current_data2 = reg_data2[j:j + chunk_size]
        # print("length of current data: ", len(current_data))
        j = j + chunk_size
        Delay = reg_dict["regular"]["2nd_channel_delay"]

        array1, array2 = generate_qkd_amplitudes(current_data1,
                                                 current_data2,
                                                 reg_dict["regular"]["data"]["case_1-0"],
                                                 reg_dict["regular"]["data"]["case_0-1"],
                                                 reg_dict["regular"]["data"]["case_1-1"])
        channel_1 = PulseSequence(reg_dict, amp_data = array1)
        channel_2 = PulseSequence(reg_dict, amp_data = array2)
        clock = PulseSequence(reg_dict, clock_sequence=True)

        channel_1.generate_times_list()
        channel_2.generate_times_list()

        channel_1.gen_pulse_objects()
        channel_2.gen_pulse_objects()

        channel_1.write_pulses()
        channel_2.write_pulses()

        # channel_1.delayChannel(1000)
        # channel_2.delayChannel(1000)

        if i == 1:
            channel_2.delay_channel(Delay, delay_info=True)
        else:
            channel_2.delay_channel(Delay)
        s1 = channel_1.plot_sequence_portion(0, 5000, title="Channel 1")
        s2 = channel_2.plot_sequence_portion(0, 5000, linking_axis=s1.x_range, title="Channel 2")
        output_file("graphs.html")
        show(column(s1, s2))

        if reg_dict["Output"]["file_save"]:
            path = reg_dict["Output"]["save_path"]
            # PulseSeq.saveSimple(path, 1, 'CH1', today_now, sequence_number = i)
            channel_1.save_csv(path, 'CH1', today_now, sequence_number=i,save_file_params=True)
            channel_2.save_csv(path, 'CH2', today_now, sequence_number=i)
            # Write out clock signals to the first two channels and the qubit
            # sequence to channels 3 and 4
            save_multi_csv([clock,clock,channel_1,channel_2],[1,2,3,4],join(path,'sequence.csv'))
            channel_1.saveParams(i)  # just save params for the first file

        # TODO
        # saving!! with bin16 and .csv


def decode_ppm(num_list,basis):
    fmt = '0' + str(basis) + 'b'
    seq = ''
    for num in num_list:
        seq = seq + format(num,fmt)
    # print(seq)
    # seq is a string of bits. Seperate it into bytes and convert to string
    return ''.join(chr(int(seq[i*8:i*8+8],2)) for i in range(len(seq)//8))


def clear_output_folder(folder):
    question = "Are you sure you want to delete the contents of " + str(folder) + " (y/n)"
    res = input(question)

    if res == 'y':
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
    else:
        return



def main():
    stream = open("params_reFormat.yaml", 'r')
    params = yaml.load(stream, Loader=yaml.FullLoader)


    divider  = 1
    laserRate = params["system"]["laser_rate"]

    if params["system"]["divider"] <= 0:
        # need to find a divider for the laser rate that results in a quotient bewteen 2.3 and 3 GHz.
        while 1:
            sync_rate = laserRate/divider
            if (sync_rate < 3) and (sync_rate > 2.3):
                # print("sync_rate is ", sync_rate, "found automatically")
                print("divider is: ", divider)
                params["system"]["divider"] = divider
                params["system"]["sync_rate"] = sync_rate
                break
            divider = divider + 1
    else:
        divider = params["system"]["divider"]
        sync_rate = params["system"]["laser_rate"]/divider
        # print("sync_rate is ", sync_rate)
        # print("divider is: ", divider, "(should be a whole number)")
        params["system"]["sync_rate"] = sync_rate

    # choose operation mode
    if params["system"]["mode"] == "ppm":
        determine_ppm_properties(params)  # this modifies the dictionary 'params' (even though it does not return anything)
        if params["ppm"]["data"]["data_source"] == "int":
            data = params["ppm"]["data"]["int_data"]
            for val in data:
                assert val <= (params["ppm"]["m_value"] - 1), "data goes from 0 to m_value - 1"
        elif params["ppm"]["data"]["data_source"] == "ext":

            # encode a string as bits, then convert the bits to PPM symbols
            data_str = params["ppm"]["data"]["str_data"]
            c = BitArray(str.encode(data_str))
            length = int(math.log2(params["ppm"]["m_value"]))
            encodable_length = (len(c.bin)//length)*length
            data = []
            for i in range(len(c.bin)//length):
                st = i*length
                data.append(int(c.bin[st:st+length],2))
            # print(data)
            #
            print("test decoding of PPM")
            print(decode_ppm(data,length))



        else:
            print("undefined data source")
        print("bitstring is this long: ", len(c))
        print("data is this long: ", len(data))
        # get output folder ready
        clear_output_folder(params["Output"]["save_path"])

        manage_ppm(params, data)

        # PulseSeq.plot_some_pulses(3)
    elif params["system"]["mode"] == "regular":

        if params["regular"]["data"]["data_source"] == "int":
            data1 = params["regular"]["data"]["int_data1"]
            data2 = params["regular"]["data"]["int_data2"]
        else:
            # TODO
            pass
        if params["regular"]["extend_sequence"]:
            determine_regular_properties(params)
        else:
            # no extension. set number of cycles to length of data
            determine_regular_properties(params, cycles=len(data1))  # sequence length, number of pulses, etc.
        clear_output_folder(params["Output"]["save_path"])
        manage_regular(params, data1, data2)
    else:
        print("unknown mode.")


main()




#
#
#
# file_name = "SWAB_1" + "_LRate_" + str(laserRate) + "GHz" + "_SRate_" + str(
#     sampleRate) + "GHz" + "_dBins_" + str(data.bins) + "_TCycles_" + str(
#     cycles_per_sequence) + "_SLength_" + str(sequenceLength) + "_dtype_" \
#             + str(data.datatype)
#
# file_name_delay = str(Delay) + "SWAB_1" + "_LRate_" + str(laserRate) + "GHz" + "_SRate_" + str(
#     sampleRate) + "GHz" + "_dBins_" + str(data.bins) + "_TCycles_" + str(
#     cycles_per_sequence) + "_SLength_" + str(sequenceLength) + "_dtype_" \
#             + str(data.datatype)
# if dict["Output"]["file_save"]:
#     with open(file_name + '.csv', 'w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerows(zip(PulseSeq.sequence, PulseSeq.zerosequence))
#
#     with open(file_name_delay + '.csv', 'w', newline='') as g:
#         writer = csv.writer(g)
#         writer.writerows(zip(PulseSeq.delaySequence, PulseSeq.zerosequence))
#
#     if dict["Output"]["clock_gen"]:
#         with open(file_name + 'CLOCK' + '.csv', 'w', newline='') as f:
#             writer = csv.writer(f)
#             writer.writerows(zip(clock_sequence, PulseSeq.zerosequence))
#
# file_name = "AWG" + "_LRate_" + str(laserRate) + "GHz" + "_SRate_" + str(
#     sampleRate) + "GHz" + "_dBins_" + str(data.bins) + "_TCycles_" + str(
#     cycles_per_sequence) + "_SLength_" + str(sequenceLength) + "_dtype_" \
#             + str(data.datatype)
#
# file_name_yaml = "AWG" + "_LRate_" + str(laserRate) + "GHz" + "_SRate_" + str(
#     sampleRate) + '.yaml'
#
#
# users = [{"LaserRate": laserRate, 'laserTime': laserTime,'seed': data.seed, 'dBins': data.bins, 'CyclesPerSequence': cycles_per_sequence, 'Pulses per Cycle': pulses_per_cycle},
#          {"pulses": pulseSequence}]
#
# with open(file_name_yaml, 'w') as f:
#     data = yaml.dump(users, f)






