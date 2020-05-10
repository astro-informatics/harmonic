
Usage
================

In main code, call lines (1) and (2) to create and initialize the logger:

	(1) import harmonic as hm 

	(2) hm.logs.setup_logging()

(note) if you wish to use a custom logging configuration simply provide the 
pathname to your yaml as the argument to setup_logging('pathname').

examples of use:

    hm.logs.low_log('a debug level message')

    hm.logs.high_log('a critical level message')


Description
================

A critical message is something that should be displayed on all runs (e.g. 
evidence value printing, run completion etc.), whereas a debug message 
should only be used for background logs to assist debugging (e.g. 
printing diagnostic checks etc.)